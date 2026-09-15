"""Materialise a local (BEIR-format) HF retrieval dataset from S3 parquet.

Reads ``corpus`` / ``queries`` / ``qrels`` parquet under an S3 prefix and writes a
local directory whose ``README.md`` declares HF ``configs`` (``corpus``,
``queries``, ``default``). MTEB's ``RetrievalDatasetLoader`` can then load it via
``load_dataset(<local_dir>, <config>)`` — i.e. point a task's ``metadata.dataset``
at the local path (see ``scripts/run_local_retrieval.py``).

Columns are kept BEIR-native (``_id`` on corpus/queries — MTEB renames to ``id`` —
and ``query-id``/``corpus-id``/``score`` on qrels).

Modes:
* ``--full``: copy every corpus parquet part (fast, no re-encode) for a full eval.
* ``--max-queries N``: build a small self-consistent subset — N queries that have
  qrels, their qrels, and a corpus of the referenced (relevant) docs plus
  ``--extra-corpus`` sampled negatives. Good for a quick end-to-end smoke test.

Example::

    python scripts/build_hf_retrieval_dataset_from_s3.py \\
        --s3-base s3://agi-prod-training-uw1-data-main-usw1/data/search/q2d-web \\
        --corpus corpus_dedup_subsampled --qrels combined \\
        --output ./local_q2d_web --max-queries 200 --extra-corpus 20000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("build_hf_retrieval_dataset_from_s3")

CORPUS_COLS = ["_id", "title", "text"]
QUERY_COLS = ["_id", "text"]
QREL_COLS = ["query-id", "corpus-id", "score"]

README_TEMPLATE = """---
configs:
- config_name: corpus
  data_files:
  - split: {split}
    path: corpus/*.parquet
- config_name: queries
  data_files:
  - split: {split}
    path: queries/*.parquet
- config_name: default
  data_files:
  - split: {split}
    path: qrels/*.parquet
---
Local BEIR-format retrieval dataset built from `{s3}`
(corpus=`{corpus}`, qrels=`{qrels}`).
"""


def _fs():
    import s3fs

    return s3fs.S3FileSystem()


def _strip(uri: str) -> str:
    return uri.replace("s3://", "").rstrip("/")


def _read_parquet(s3_glob: str, columns: list[str]):
    """Read a parquet file/dir from S3 into a pyarrow Table (select ``columns``)."""
    import pyarrow.dataset as pds

    fs = _fs()
    dataset = pds.dataset(_strip(s3_glob), filesystem=fs, format="parquet")
    return dataset.to_table(columns=columns)


def _write_table(table, out_dir: Path, name: str) -> None:
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / f"{name}.parquet")


def _copy_corpus_full(s3_corpus: str, out_dir: Path) -> int:
    """Copy every corpus parquet part verbatim (no re-encode)."""
    import pyarrow.parquet as pq

    fs = _fs()
    parts = [p for p in fs.ls(_strip(s3_corpus)) if p.endswith(".parquet")]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, part in enumerate(parts):
        with fs.open(part, "rb") as f:
            fs.get(part, str(out_dir / f"part-{i:05d}.parquet"))
        n += pq.read_metadata(str(out_dir / f"part-{i:05d}.parquet")).num_rows
        logger.info("copied corpus part %d/%d (%s rows)", i + 1, len(parts), n)
    return n


def _fetch_corpus_by_ids(s3_corpus: str, ids: set[str], extra: int):
    """Fetch corpus rows whose ``_id`` is in ``ids``, plus ``extra`` sampled negatives."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pds

    fs = _fs()
    dataset = pds.dataset(_strip(s3_corpus), filesystem=fs, format="parquet")
    id_list = pa.array(list(ids), type=pa.string())

    wanted = dataset.to_table(
        columns=CORPUS_COLS, filter=pc.field("_id").isin(id_list)
    )
    logger.info("fetched %d relevant corpus docs (of %d ids)", wanted.num_rows, len(ids))

    if extra > 0:
        collected, neg_tables = 0, []
        for batch in dataset.to_batches(columns=CORPUS_COLS, batch_size=100_000):
            tbl = pa.Table.from_batches([batch])
            tbl = tbl.filter(pc.invert(pc.is_in(tbl.column("_id"), value_set=id_list)))
            if collected + tbl.num_rows > extra:
                tbl = tbl.slice(0, extra - collected)
            neg_tables.append(tbl)
            collected += tbl.num_rows
            if collected >= extra:
                break
        if neg_tables:
            wanted = pa.concat_tables([wanted, *neg_tables])
        logger.info("added %d negative corpus docs -> %d total", collected, wanted.num_rows)
    return wanted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--s3-base",
        default="s3://agi-prod-training-uw1-data-main-usw1/data/search/q2d-web",
    )
    p.add_argument("--corpus", default="corpus_dedup_subsampled")
    p.add_argument("--qrels", default="combined")
    p.add_argument("--queries-file", default="queries/queries.parquet")
    p.add_argument("--output", required=True, help="Local output directory.")
    p.add_argument("--split", default="test")
    p.add_argument("--full", action="store_true", help="Copy the entire corpus.")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--extra-corpus", type=int, default=20_000)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    out = Path(args.output)
    s3_corpus = f"{args.s3_base}/{args.corpus}"
    s3_qrels = f"{args.s3_base}/{args.qrels}" if "/" in args.qrels else f"{args.s3_base}/qrels/{args.qrels}"

    logger.info("Reading queries + qrels ...")
    queries = _read_parquet(f"{args.s3_base}/{args.queries_file}", QUERY_COLS)
    qrels = _read_parquet(s3_qrels, QREL_COLS)

    if not args.full:
        import pyarrow as pa
        import pyarrow.compute as pc

        q_with_qrels = set(qrels.column("query-id").to_pylist())
        keep_qids = [q for q in queries.column("_id").to_pylist() if q in q_with_qrels]
        keep_qids = keep_qids[: args.max_queries]
        qid_set = set(keep_qids)
        queries = queries.filter(pc.field("_id").isin(pa.array(keep_qids, pa.string())))
        qrels = qrels.filter(pc.field("query-id").isin(pa.array(keep_qids, pa.string())))
        rel_ids = set(qrels.column("corpus-id").to_pylist())
        logger.info(
            "subset: %d queries, %d qrels, %d relevant docs",
            len(qid_set), qrels.num_rows, len(rel_ids),
        )
        corpus = _fetch_corpus_by_ids(s3_corpus, rel_ids, args.extra_corpus)
        _write_table(corpus, out / "corpus", "corpus")
        n_corpus = corpus.num_rows
    else:
        n_corpus = _copy_corpus_full(s3_corpus, out / "corpus")

    _write_table(queries, out / "queries", "queries")
    _write_table(qrels, out / "qrels", "qrels")

    (out / "README.md").write_text(
        README_TEMPLATE.format(
            split=args.split, s3=args.s3_base, corpus=args.corpus, qrels=args.qrels
        )
    )

    logger.info(
        "Wrote local dataset to %s (corpus=%d, queries=%d, qrels=%d)",
        out, n_corpus, queries.num_rows, qrels.num_rows,
    )
    print(f"LOCAL_DATASET {out.resolve()}")


if __name__ == "__main__":
    main()
