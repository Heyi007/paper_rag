"""Stage 1 — build / load the 10k paper-abstract corpus.

Primary source: the big arXiv-metadata parquet shards of
`librarian-bots/arxiv_abstracts`, read over ranged HTTP (HfFileSystem +
pyarrow) so we never download a full 280 MB shard. CS / stat.ML papers are
preferred because the query set is ML-focused.

A local `.jsonl` / `.parquet` file can replace the remote source via
`--from-file`, which also keeps the pipeline runnable fully offline.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

from . import config as cfg
from .util import banner, normalize_ws, read_jsonl, show_artifact, write_jsonl

ARXIV_URL = "https://arxiv.org/abs/{arxiv_id}"


def _make_doc(idx: int, *, arxiv_id: str, title: str, abstract: str,
              url: str | None, categories: str) -> dict[str, Any] | None:
    title = normalize_ws(title)
    abstract = normalize_ws(abstract)
    arxiv_id = normalize_ws(arxiv_id)
    if not title or not abstract or not arxiv_id:
        return None
    return {
        "doc_id": f"D{idx:05d}",
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "url": url or ARXIV_URL.format(arxiv_id=arxiv_id),
        "categories": normalize_ws(categories),
    }


def _is_csml(categories: str) -> bool:
    return any(
        tok.startswith(prefix)
        for prefix in cfg.CS_PREFIXES
        for tok in (categories or "").split()
    )


def _iter_remote_rows() -> Iterator[dict[str, Any]]:
    from .util import sanitize_proxy_env

    sanitize_proxy_env()  # idempotent; protects direct module usage as well
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    files: list[str] = []
    for f in cfg.HF_SHARD_FILES:
        if "*" in f:
            files.extend(sorted(fs.glob(f"hf://datasets/{cfg.HF_DATASET_REPO}/{f}")))
        else:
            files.append(f"hf://datasets/{cfg.HF_DATASET_REPO}/{f}")
    row_groups_seen = 0
    for path in files:
        pf = pq.ParquetFile(fs.open(path, "rb"))
        for rg in range(pf.num_row_groups):
            if row_groups_seen >= cfg.MAX_ROW_GROUPS:
                return
            table = pf.read_row_group(
                rg, columns=["id", "title", "abstract", "categories"]
            )
            row_groups_seen += 1
            for row in table.to_pylist():
                yield row


def _iter_local_rows(path: str) -> Iterator[dict[str, Any]]:
    if path.endswith(".jsonl"):
        for row in read_jsonl(path):
            yield row
        return
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for row in table.to_pylist():
        yield row


def _doc_from_local_row(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    arxiv_id = row.get("arxiv_id") or row.get("id") or row.get("doc_id") or f"local{idx:05d}"
    return _make_doc(
        idx,
        arxiv_id=str(arxiv_id),
        title=str(row.get("title", "")),
        abstract=str(row.get("abstract", "")),
        url=row.get("url"),
        categories=str(row.get("categories") or row.get("category") or ""),
    )


def prepare_corpus(config: cfg.PipelineConfig) -> list[dict[str, Any]]:
    """Return the corpus as a list of normalized docs, cached as JSONL."""
    os.makedirs(config.data_dir, exist_ok=True)
    if not config.rebuild and os.path.exists(config.corpus_path):
        docs = read_jsonl(config.corpus_path)
        if len(docs) >= config.n_docs:
            banner(f"STAGE 1/5  DATA (cached: {config.corpus_path})")
            print(f"loaded {len(docs[: config.n_docs])} cached docs", flush=True)
            return docs[: config.n_docs]

    banner("STAGE 1/5  DATA — prepare paper abstract corpus")
    source = config.from_file or f"hf://{cfg.HF_DATASET_REPO} (cs./stat.ML preferred)"
    print(f"source: {source}", flush=True)

    row_iter = _iter_local_rows(config.from_file) if config.from_file else _iter_remote_rows()
    docs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    backfill: list[dict[str, Any]] = []

    for row in row_iter:
        if not config.from_file:
            row = {
                "arxiv_id": row.get("id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "categories": row.get("categories"),
            }
        categories = str(row.get("categories") or "")
        doc = _doc_from_local_row(row, len(docs) + len(backfill))
        if doc is None or doc["arxiv_id"] in seen_ids:
            continue
        seen_ids.add(doc["arxiv_id"])
        if config.from_file or _is_csml(categories):
            doc["doc_id"] = f"D{len(docs):05d}"
            docs.append(doc)
            if len(docs) >= config.n_docs:
                break
        else:
            backfill.append(doc)

    # Remote source safety net: if the CS filter under-fills the row-group
    # budget, top up with general arXiv papers rather than failing the run.
    if len(docs) < config.n_docs and not config.from_file:
        for doc in backfill:
            doc["doc_id"] = f"D{len(docs):05d}"
            docs.append(doc)
            if len(docs) >= config.n_docs:
                break
    if len(docs) < config.n_docs:
        raise RuntimeError(
            f"only {len(docs)} docs available, need {config.n_docs}; "
            "lower --docs or provide another --from-file"
        )

    write_jsonl(config.corpus_path, docs)
    meta = {
        "source": source,
        "n_docs": len(docs),
        "columns": list(docs[0].keys()),
        "corpus_path": config.corpus_path,
    }
    with open(os.path.join(config.data_dir, "corpus.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(docs)} docs -> {config.corpus_path}", flush=True)
    show_artifact("corpus sample (first doc)", docs[0], config.print_full)
    return docs
