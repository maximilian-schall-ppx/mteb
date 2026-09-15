"""Tests for the multi-node ``DistributedEncoderWrapper``.

The collective tests spin up a small gloo (CPU) process group via
``torch.multiprocessing.spawn`` so they run without GPUs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from mteb.models.distributed_wrapper import (
    DistributedEncoderWrapper,
    DistributedSearchWrapper,
    _all_gather_array,
    _block_bounds,
    _merge_partial_results,
)


@pytest.mark.parametrize(
    ("n", "world_size", "expected"),
    [
        (10, 2, [(0, 5), (5, 10)]),
        (10, 3, [(0, 4), (4, 7), (7, 10)]),  # remainder to the first ranks
        (5, 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        (2, 5, [(0, 1), (1, 2), (2, 2), (2, 2), (2, 2)]),  # n < world_size
        (0, 3, [(0, 0), (0, 0), (0, 0)]),
    ],
)
def test_block_bounds(n, world_size, expected):
    bounds = _block_bounds(n, world_size)
    assert bounds == expected
    # Blocks tile range(n) contiguously with no gaps or overlaps.
    covered = [i for start, end in bounds for i in range(start, end)]
    assert covered == list(range(n))


class _ToyDataset:
    """Map-style dataset whose row ``i`` is ``{"val": i}``."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"val": idx}


class _ToyEncoder:
    """Encodes row ``i`` to the constant vector ``[i, i, i]`` (dim=3)."""

    mteb_model_meta = "toy-meta"

    def encode(self, inputs: DataLoader, **kwargs: object) -> np.ndarray:
        vals: list[int] = []
        for batch in inputs:
            vals.extend(int(v) for v in batch["val"])
        return np.array([[v, v, v] for v in vals], dtype=np.float32)

    def similarity(self, a, b):
        return a @ b.T

    def similarity_pairwise(self, a, b):
        return (a * b).sum(-1)


def _free_port() -> int:
    """Pick an OS-assigned free TCP port for the rendezvous.

    Chosen by the parent and passed to every worker so concurrent gloo groups
    (or fast teardown/re-init) never collide on a fixed port.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_pg(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _all_gather_worker(
    rank: int, port: int, world_size: int, n: int, dim: int, q
) -> None:
    _init_pg(rank, world_size, port)
    try:
        counts = [end - start for start, end in _block_bounds(n, world_size)]
        start, end = _block_bounds(n, world_size)[rank]
        # Local block: row i -> [i]*dim.
        local = np.array(
            [[i] * dim for i in range(start, end)], dtype=np.float32
        ).reshape(end - start, dim)
        full = _all_gather_array(local, counts, rank, world_size)
        if rank == 0:
            q.put(full)
    finally:
        dist.destroy_process_group()


def _wrapper_worker(rank: int, port: int, world_size: int, n: int, q) -> None:
    _init_pg(rank, world_size, port)
    try:
        loader = DataLoader(_ToyDataset(n), batch_size=4, shuffle=False)
        wrapper = DistributedEncoderWrapper(_ToyEncoder())
        out = wrapper.encode(
            loader,
            task_metadata=_FakeMeta(),
            hf_split="test",
            hf_subset="default",
            prompt_type=None,
        )
        if rank == 0:
            q.put(np.asarray(out))
    finally:
        dist.destroy_process_group()


class _FakeMeta:
    name = "toy-task"


def _run_workers(fn, world_size: int, extra: tuple, timeout: int = 60):
    """Spawn ``world_size`` gloo workers on a fresh free port; return rank 0's result.

    Workers receive ``(rank, port, world_size, *extra, queue)``.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(
        fn,
        args=(_free_port(), world_size, *extra, q),
        nprocs=world_size,
        join=True,
    )
    return q.get(timeout=timeout)


@pytest.mark.parametrize(("n", "world_size"), [(10, 2), (7, 3), (2, 4)])
def test_all_gather_array_restores_order(n, world_size):
    full = _run_workers(_all_gather_worker, world_size, (n, 3))
    expected = np.array([[i, i, i] for i in range(n)], dtype=np.float32)
    np.testing.assert_array_equal(full, expected)


@pytest.mark.parametrize(("n", "world_size"), [(10, 2), (9, 3)])
def test_distributed_wrapper_encode_matches_single_process(n, world_size):
    out = _run_workers(_wrapper_worker, world_size, (n,))

    # Reference: the same toy encoder run on the full dataset in one process.
    reference = _ToyEncoder().encode(DataLoader(_ToyDataset(n), batch_size=4))
    np.testing.assert_array_equal(out, reference)


def test_wrapper_is_encoder_not_search_protocol():
    from mteb.models.models_protocols import EncoderProtocol, SearchProtocol

    wrapper = DistributedEncoderWrapper(_ToyEncoder())
    # Must look like an encoder (so retrieval wraps it in SearchEncoderWrapper)
    # but not a search model (which would bypass that wrapping).
    assert isinstance(wrapper, EncoderProtocol)
    assert not isinstance(wrapper, SearchProtocol)


def test_merge_partial_results_full_corpus():
    """Disjoint per-shard top-k results merge to the global top-k per query."""
    # Two ranks, each returning its shard's top-2 for the same two queries.
    part0 = {"q1": {"d0": 0.1, "d1": 0.9}, "q2": {"d0": 0.5, "d1": 0.2}}
    part1 = {"q1": {"d2": 0.8, "d3": 0.3}, "q2": {"d2": 0.7, "d3": 0.6}}
    merged = _merge_partial_results([part0, part1], top_k=2)
    assert set(merged["q1"]) == {"d1", "d2"}  # 0.9, 0.8
    assert set(merged["q2"]) == {"d2", "d3"}  # 0.7, 0.6


def test_merge_partial_results_query_sharded():
    """Disjoint per-query results (rerank fallback) merge by union with no loss."""
    part0 = {"q1": {"d0": 0.9, "d1": 0.5}}
    part1 = {"q2": {"d2": 0.8, "d3": 0.4}}
    merged = _merge_partial_results([part0, part1], top_k=5)
    assert merged == {**part0, **part1}


def test_search_wrapper_is_search_not_encoder_protocol():
    from mteb.models.models_protocols import EncoderProtocol, SearchProtocol

    wrapper = DistributedSearchWrapper(_ToyEncoder())
    # Must look like a search model (used directly by retrieval) and not an
    # encoder (which would be re-wrapped in SearchEncoderWrapper).
    assert isinstance(wrapper, SearchProtocol)
    assert not isinstance(wrapper, EncoderProtocol)


class _ToyDistSearch(DistributedSearchWrapper):
    """Toy search wrapper: scores docs by a ``val`` column, ignoring the model.

    Lets us exercise corpus sharding + all-gather + merge without a real encoder.
    """

    def _local_search(self, corpus, queries, *, top_k, top_ranked=None, **kw):  # noqa: ANN003
        ranked = sorted(
            zip(corpus["id"], corpus["val"], strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return {
            qid: {cid: float(v) for cid, v in ranked} for qid in queries["id"]
        }


def _toy_corpus_queries(n_corpus: int, n_queries: int):
    from datasets import Dataset

    corpus = Dataset.from_dict(
        {"id": [f"d{i}" for i in range(n_corpus)], "val": list(range(n_corpus))}
    )
    queries = Dataset.from_dict({"id": [f"q{i}" for i in range(n_queries)]})
    return corpus, queries


def _search_worker(
    rank: int, port: int, world_size: int, n_corpus: int, top_k: int, q
) -> None:
    _init_pg(rank, world_size, port)
    try:
        corpus, queries = _toy_corpus_queries(n_corpus, n_queries=3)
        wrapper = _ToyDistSearch(_ToyEncoder())
        wrapper.index(
            corpus,
            task_metadata=_FakeMeta(),
            hf_split="test",
            hf_subset="default",
            encode_kwargs={},
        )
        result = wrapper.search(
            queries,
            task_metadata=_FakeMeta(),
            hf_split="test",
            hf_subset="default",
            top_k=top_k,
            encode_kwargs={},
        )
        if rank == 0:
            q.put(result)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("n_corpus", "world_size"), [(10, 2), (7, 3)])
def test_distributed_search_matches_single_process(n_corpus, world_size):
    top_k = 3
    result = _run_workers(_search_worker, world_size, (n_corpus, top_k))

    # Reference: highest `val` doc ids overall are the global top-k.
    expected_ids = {f"d{i}" for i in range(n_corpus - top_k, n_corpus)}
    assert set(result) == {"q0", "q1", "q2"}
    for docs in result.values():
        assert set(docs) == expected_ids


class _ToySearchModel:
    """A minimal ``SearchProtocol`` model (stands in for ColBERT / BM25).

    Scores each doc by its ``val`` column, independent of the query, so the
    global top-k is deterministic and easy to assert.
    """

    mteb_model_meta = "toy-search-meta"

    def __init__(self) -> None:
        self._corpus = None

    def index(self, corpus, **kw) -> None:  # noqa: ANN003
        self._corpus = corpus

    def search(self, queries, *, top_k, top_ranked=None, **kw):  # noqa: ANN003
        ranked = sorted(
            zip(self._corpus["id"], self._corpus["val"], strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return {
            qid: {cid: float(v) for cid, v in ranked} for qid in queries["id"]
        }


def test_search_wrapper_delegates_to_searchprotocol_without_pg():
    """A SearchProtocol model is used directly (not wrapped in SearchEncoderWrapper)."""
    from mteb.models.models_protocols import SearchProtocol

    assert not dist.is_initialized()
    corpus, queries = _toy_corpus_queries(n_corpus=8, n_queries=2)
    model = _ToySearchModel()
    assert isinstance(model, SearchProtocol)

    wrapper = DistributedSearchWrapper(model)
    wrapper.index(
        corpus,
        task_metadata=_FakeMeta(),
        hf_split="test",
        hf_subset="default",
        encode_kwargs={},
    )
    result = wrapper.search(
        queries,
        task_metadata=_FakeMeta(),
        hf_split="test",
        hf_subset="default",
        top_k=3,
        encode_kwargs={},
    )
    expected_ids = {"d5", "d6", "d7"}  # top-3 vals
    assert set(result) == {"q0", "q1"}
    for docs in result.values():
        assert set(docs) == expected_ids


def _dist_search_model_worker(
    rank: int,
    port: int,
    world_size: int,
    n_corpus: int,
    n_queries: int,
    top_k: int,
    shard: str,
    q,
) -> None:
    _init_pg(rank, world_size, port)
    try:
        corpus, queries = _toy_corpus_queries(n_corpus, n_queries=n_queries)
        wrapper = DistributedSearchWrapper(_ToySearchModel(), shard=shard)
        wrapper.index(
            corpus,
            task_metadata=_FakeMeta(),
            hf_split="test",
            hf_subset="default",
            encode_kwargs={},
        )
        result = wrapper.search(
            queries,
            task_metadata=_FakeMeta(),
            hf_split="test",
            hf_subset="default",
            top_k=top_k,
            encode_kwargs={},
        )
        if rank == 0:
            q.put(result)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("shard", "n_corpus", "n_queries", "world_size"),
    [
        ("corpus", 10, 3, 2),  # late-interaction-style: shard corpus, merge top-k
        ("query", 8, 5, 3),  # BM25-style: full index per rank, shard queries
    ],
)
def test_distributed_searchprotocol_matches_single_process(
    shard, n_corpus, n_queries, world_size
):
    top_k = 3
    result = _run_workers(
        _dist_search_model_worker, world_size, (n_corpus, n_queries, top_k, shard)
    )

    expected_ids = {f"d{i}" for i in range(n_corpus - top_k, n_corpus)}
    assert set(result) == {f"q{i}" for i in range(n_queries)}
    for docs in result.values():
        assert set(docs) == expected_ids


def test_encode_delegates_without_process_group():
    """With no active process group, encode must equal the wrapped model exactly."""
    assert not dist.is_initialized()
    wrapper = DistributedEncoderWrapper(_ToyEncoder())
    loader = DataLoader(_ToyDataset(6), batch_size=4, shuffle=False)
    out = wrapper.encode(
        loader,
        task_metadata=_FakeMeta(),
        hf_split="test",
        hf_subset="default",
        prompt_type=None,
    )
    reference = _ToyEncoder().encode(DataLoader(_ToyDataset(6), batch_size=4))
    np.testing.assert_array_equal(np.asarray(out), reference)
