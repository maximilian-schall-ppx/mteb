from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.model_meta import ModelMeta
    from mteb.types import (
        Array,
        BatchedInput,
        CorpusDatasetType,
        EncodeKwargs,
        PromptType,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

    from .models_protocols import EncoderProtocol

logger = logging.getLogger(__name__)


def _block_bounds(n: int, world_size: int) -> list[tuple[int, int]]:
    """Partition ``range(n)`` into ``world_size`` contiguous blocks.

    Block sizes differ by at most one, so concatenating them in rank order
    restores the original order.

    Args:
        n: Number of items to partition.
        world_size: Number of ranks to partition across.

    Returns:
        One ``(start, end)`` tuple per rank, covering ``range(n)`` exactly.
    """
    base, rem = divmod(n, world_size)
    bounds: list[tuple[int, int]] = []
    start = 0
    for r in range(world_size):
        count = base + (1 if r < rem else 0)
        bounds.append((start, start + count))
        start += count
    return bounds


def _to_numpy(embeddings: Array) -> np.ndarray:
    """Convert an encoder's output to a CPU float32 numpy array."""
    import numpy as np

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=np.float32)


def _is_distributed() -> bool:
    """Whether a torch.distributed process group with >1 rank is active."""
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _collective_device() -> torch.device:
    """Device to run the collective on: CUDA for the NCCL backend, else CPU."""
    import torch
    import torch.distributed as dist

    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _all_gather_array(
    local: np.ndarray,
    counts: list[int],
    rank: int,
    world_size: int,
) -> np.ndarray:
    """All-gather per-rank embedding blocks into the full array on every rank.

    Blocks are padded to the max size so ``all_gather`` sees equal shapes, then
    sliced back to their known counts and concatenated in rank order.

    Args:
        local: This rank's ``(counts[rank], dim)`` embeddings (possibly 0 rows).
        counts: Row count contributed by each rank (same on all ranks).
        rank: This process's rank.
        world_size: Total number of ranks.

    Returns:
        The full ``(sum(counts), dim)`` float32 numpy array, identical on all ranks.
    """
    import numpy as np
    import torch
    import torch.distributed as dist

    device = _collective_device()

    # Ranks with 0 rows do not know the embedding dim; agree on it via all-reduce.
    local_dim = int(local.shape[1]) if local.ndim == 2 and local.shape[0] > 0 else 0
    dim_t = torch.tensor([local_dim], device=device, dtype=torch.int64)
    dist.all_reduce(dim_t, op=dist.ReduceOp.MAX)
    dim = int(dim_t.item())
    if dim == 0:
        return np.zeros((0, 0), dtype=np.float32)

    max_count = max(counts)
    padded = torch.zeros((max_count, dim), device=device, dtype=torch.float32)
    if counts[rank] > 0:
        padded[: counts[rank]] = torch.as_tensor(
            local, device=device, dtype=torch.float32
        )

    gathered = [
        torch.zeros((max_count, dim), device=device, dtype=torch.float32)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded)

    parts = [gathered[r][: counts[r]] for r in range(world_size)]
    return torch.cat(parts, dim=0).cpu().numpy()


class DistributedEncoderWrapper:
    """Shards ``encode`` work across ``torch.distributed`` ranks.

    Wraps any [EncoderProtocol][mteb.models.EncoderProtocol]. Each rank encodes a
    contiguous slice of the inputs, then all ranks ``all_gather`` the partial
    embeddings to reconstruct the full ``(N, dim)`` array in original order.
    Implements ``EncoderProtocol`` but not ``SearchProtocol``, so retrieval still
    wraps it in ``SearchEncoderWrapper``. With no active process group, ``encode``
    delegates to the wrapped model (exact single-process equivalence).

    Args:
        model: The already-loaded encoder to distribute (on this rank's device).
    """

    def __init__(self, model: EncoderProtocol) -> None:
        self.model = model

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Metadata of the wrapped model."""
        return self.model.mteb_model_meta

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Encode ``inputs``, sharding the work across ranks and gathering results."""
        if not _is_distributed():
            return self.model.encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )

        import numpy as np
        import torch.distributed as dist
        from torch.utils.data import DataLoader, Subset

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        dataset = inputs.dataset
        n = len(dataset)  # type: ignore[arg-type]
        counts = [end - start for start, end in _block_bounds(n, world_size)]
        start, end = _block_bounds(n, world_size)[rank]

        logger.info(
            "Rank %d/%d encoding rows [%d:%d) of %d for %s (%s)",
            rank,
            world_size,
            start,
            end,
            n,
            task_metadata.name,
            prompt_type,
        )

        # Rebuild a loader over this rank's slice, keeping the original collate/batching.
        sub_loader = DataLoader(
            Subset(dataset, range(start, end)),
            batch_size=inputs.batch_size,
            collate_fn=inputs.collate_fn,
            num_workers=inputs.num_workers,
            shuffle=False,
        )

        local_kwargs: dict[str, Any] = dict(kwargs)
        if rank != 0:
            local_kwargs["show_progress_bar"] = False

        if end > start:
            local = _to_numpy(
                self.model.encode(
                    sub_loader,
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    prompt_type=prompt_type,
                    **local_kwargs,
                )
            )
        else:
            local = np.zeros((0, 0), dtype=np.float32)

        return _all_gather_array(local, counts, rank, world_size)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute the similarity between two collections of embeddings."""
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute the pairwise similarity between two collections of embeddings."""
        return self.model.similarity_pairwise(embeddings1, embeddings2)


def _merge_partial_results(
    partials: list[RetrievalOutputType], top_k: int
) -> RetrievalOutputType:
    """Merge per-rank ``{qid: {doc_id: score}}`` results, keeping global top-k.

    Args:
        partials: One result dict per rank (in rank order).
        top_k: Number of documents to keep per query.

    Returns:
        The merged results, identical on every rank.
    """
    import heapq

    merged: RetrievalOutputType = {}
    for part in partials:
        for qid, docs in part.items():
            merged.setdefault(qid, {}).update(docs)
    for qid, docs in merged.items():
        if len(docs) > top_k:
            merged[qid] = dict(
                heapq.nlargest(top_k, docs.items(), key=lambda kv: kv[1])
            )
    return merged


class DistributedSearchWrapper:
    """Shards retrieval search across ``torch.distributed`` ranks.

    Implements [SearchProtocol][mteb.models.SearchProtocol], so retrieval tasks
    use it directly. Dense encoders are wrapped per rank in ``SearchEncoderWrapper``;
    ``SearchProtocol``-native models (ColBERT/PyLate, BM25) are delegated to. With
    no active process group, ``search`` runs a single local search.

    Sharding modes:

    * ``"corpus"`` (default): each rank indexes and scores a disjoint corpus shard;
      partial per-query top-k are gathered and merged. Valid when a (query, doc)
      score is corpus-independent — dense (cosine/dot) and late-interaction (MaxSim).
    * ``"query"``: each rank builds the full index and searches a disjoint query
      slice; results are unioned. Required for BM25, whose scores depend on global
      corpus statistics (IDF, average doc length). Also used for reranking.

    Args:
        model: The model to distribute (dense encoder or ``SearchProtocol``).
        corpus_chunk_size: Chunk size for the per-rank ``SearchEncoderWrapper``.
        shard: ``"corpus"`` (dense, late-interaction) or ``"query"`` (BM25).
    """

    task_corpus: CorpusDatasetType | None

    def __init__(
        self,
        model: EncoderProtocol,
        corpus_chunk_size: int = 50_000,
        shard: str = "corpus",
    ) -> None:
        if shard not in {"corpus", "query"}:
            raise ValueError(f"shard must be 'corpus' or 'query', got {shard!r}")
        self.model = model
        self.corpus_chunk_size = corpus_chunk_size
        self.shard = shard
        self.task_corpus = None

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Metadata of the wrapped model."""
        return self.model.mteb_model_meta

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        """Retain the corpus; encoding happens (sharded) during ``search``."""
        self.task_corpus = corpus

    def _local_search(
        self,
        corpus: CorpusDatasetType,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        """Run a standard (non-distributed) search over ``corpus`` on this rank.

        ``SearchProtocol``-native models own their index/scoring and are used
        directly; dense encoders are wrapped in ``SearchEncoderWrapper``.
        """
        from .models_protocols import SearchProtocol
        from .search_wrappers import SearchEncoderWrapper

        local = (
            self.model
            if isinstance(self.model, SearchProtocol)
            else SearchEncoderWrapper(
                self.model, corpus_chunk_size=self.corpus_chunk_size
            )
        )
        local.index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )
        return local.search(
            queries,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            top_ranked=top_ranked,
            num_proc=num_proc,
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        """Search the corpus, sharding the work across ranks and merging results."""
        if self.task_corpus is None:
            raise ValueError("Corpus must be indexed before searching.")

        common = dict(
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            top_k=top_k,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

        if not _is_distributed():
            return self._local_search(
                self.task_corpus, queries, top_ranked=top_ranked, **common
            )

        import torch.distributed as dist

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Query-sharding is required for BM25 (global corpus stats) and reranking.
        if self.shard == "query" or top_ranked is not None:
            return self._query_sharded_search(
                queries,
                corpus=self.task_corpus,
                rank=rank,
                world_size=world_size,
                top_ranked=top_ranked,
                **common,
            )

        start, end = _block_bounds(len(self.task_corpus), world_size)[rank]
        shard = self.task_corpus.select(range(start, end))
        logger.info(
            "Rank %d/%d searching corpus shard [%d:%d) of %d for %s",
            rank,
            world_size,
            start,
            end,
            len(self.task_corpus),
            task_metadata.name,
        )
        partial = self._local_search(shard, queries, top_ranked=None, **common)

        gathered: list[RetrievalOutputType] = [{} for _ in range(world_size)]
        dist.all_gather_object(gathered, partial)
        return _merge_partial_results(gathered, top_k)

    def _query_sharded_search(  # noqa: PLR0913
        self,
        queries: QueryDatasetType,
        *,
        corpus: CorpusDatasetType,
        rank: int,
        world_size: int,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        """Each rank searches a disjoint slice of queries against the full corpus."""
        import torch.distributed as dist

        start, end = _block_bounds(len(queries), world_size)[rank]
        q_shard = queries.select(range(start, end))
        logger.info(
            "Rank %d/%d searching queries [%d:%d) of %d for %s",
            rank,
            world_size,
            start,
            end,
            len(queries),
            task_metadata.name,
        )

        partial: RetrievalOutputType
        if len(q_shard) == 0:
            partial = {}
        else:
            tr = None
            if top_ranked is not None:
                shard_ids = set(q_shard["id"])
                tr = {q: d for q, d in top_ranked.items() if q in shard_ids}
            partial = self._local_search(
                corpus,
                q_shard,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                top_k=top_k,
                encode_kwargs=encode_kwargs,
                top_ranked=tr,
                num_proc=num_proc,
            )

        gathered: list[RetrievalOutputType] = [{} for _ in range(world_size)]
        dist.all_gather_object(gathered, partial)
        return _merge_partial_results(gathered, top_k)
