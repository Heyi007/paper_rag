#!/usr/bin/env python3
"""paper_rag entry point.

Fixed pipeline (no rewriting, no agentic loop, no tuning):

  10k arXiv abstracts
      -> BM25 top50 + BGE dense top50 -> RRF fused top50
      -> cross-encoder rerank -> top5
      -> cited prompt -> LLM answer with inline [n] citations
      -> clickable HTML report + every intermediate artifact on disk

Example:
  python run.py                      # full 30-query run on a single GPU
  python run.py --llm stub --docs 500 --query-limit 5 --from-file mini.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Defaults for the corporate network; overridden silently if already exported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "http://huggingface-proxy-sg.byted.org")

# Must run before huggingface_hub/httpx build any client: some worker images
# ship malformed NO_PROXY/proxy vars that crash httpx at construction time.
from paper_rag.util import sanitize_proxy_env  # noqa: E402

sanitize_proxy_env()

from paper_rag import config as cfg  # noqa: E402
from paper_rag.data import prepare_corpus  # noqa: E402
from paper_rag.generation import (  # noqa: E402
    answer_query,
    make_generator,
    print_generation_banner,
)
from paper_rag.rerank import (  # noqa: E402
    CrossEncoderReranker,
    print_rerank_banner,
    rerank,
)
from paper_rag.report import write_html_report  # noqa: E402
from paper_rag.retrieval import Retriever, print_retrieval_banner  # noqa: E402
from paper_rag.util import (  # noqa: E402
    Timer,
    banner,
    read_jsonl,
    resolve_device,
    show_artifact,
    write_jsonl,
)


def parse_args() -> cfg.PipelineConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", type=int, default=cfg.N_DOCS, help="corpus size (default 10000)")
    p.add_argument("--query-limit", type=int, default=30)
    p.add_argument("--from-file", help="local .jsonl/.parquet corpus instead of Hugging Face")
    p.add_argument("--queries-file", default=cfg.DEFAULT_QUERIES_FILE)
    p.add_argument("--llm", choices=["hf", "openai", "stub"], default="hf")
    p.add_argument("--llm-model", default=cfg.DEFAULT_LLM_ID)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--data-dir", default=cfg.DEFAULT_DATA_DIR)
    p.add_argument("--artifacts-dir", default=cfg.DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--print-full", action="store_true", help="print untruncated artifacts to console")
    p.add_argument("--rebuild", action="store_true", help="ignore cached corpus/embeddings")
    args = p.parse_args()
    return cfg.PipelineConfig(
        n_docs=args.docs,
        from_file=args.from_file,
        queries_file=args.queries_file,
        query_limit=args.query_limit,
        llm_backend=args.llm,
        llm_model_id=args.llm_model,
        device=args.device,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        print_full=args.print_full,
        rebuild=args.rebuild,
    )


def main() -> int:
    config = parse_args()
    device = resolve_device(config.device)
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.artifacts_dir, run_id)
    prompt_dir = os.path.join(run_dir, "04_prompts")
    os.makedirs(run_dir, exist_ok=True)
    timer = Timer()

    print(f"run dir: {run_dir}")
    print(f"device: {device} | llm backend: {config.llm_backend} ({config.llm_model_id})")

    # ---- Stage 1: corpus ----------------------------------------------------
    docs = prepare_corpus(config)
    docs_by_id = {d["doc_id"]: d for d in docs}
    timer.stage("1_data")

    # ---- Stage 2: dual-path retrieval --------------------------------------
    print_retrieval_banner()
    retriever = Retriever(config, docs, device)
    timer.stage("2_index_build")
    queries = read_jsonl(config.queries_file)[: config.query_limit]
    retrieval_results = []
    for i, q in enumerate(queries):
        result = retriever.search(q["qid"], q["question"], verbose=(i == 0 or config.print_full))
        retrieval_results.append(result)
        if i != 0 and not config.print_full:
            top = ", ".join(f"{h['doc_id']}" for h in result["fused_top50"][:5])
            print(f"  [{q['qid']}] fused top5: {top}", flush=True)
    write_jsonl(os.path.join(run_dir, "02_retrieval.jsonl"), retrieval_results)
    timer.stage("2_retrieval_all_queries")

    # ---- Stage 3: cross-encoder rerank -------------------------------------
    print_rerank_banner()
    reranker = CrossEncoderReranker(
        config.rerank_model_id, device, cfg.RERANK_BATCH, cfg.RERANK_MAX_LEN
    )
    rerank_results = []
    for i, retr in enumerate(retrieval_results):
        result = rerank(config, reranker, docs_by_id, retr, verbose=(i == 0 or config.print_full))
        rerank_results.append(result)
        if i != 0 and not config.print_full:
            top = ", ".join(
                f"{h['doc_id']}({h['rerank_score']})" for h in result["top5"]
            )
            print(f"  [{retr['qid']}] rerank top5: {top}", flush=True)
    write_jsonl(os.path.join(run_dir, "03_rerank.jsonl"), rerank_results)
    timer.stage("3_rerank_all_queries")

    # ---- Stages 4-5: prompt + generation -----------------------------------
    print_generation_banner()
    generator = make_generator(config, device)
    answers = []
    for i, rer in enumerate(rerank_results):
        result = answer_query(
            config, generator, docs_by_id, rer, prompt_dir,
            verbose=(i == 0 or config.print_full),
        )
        answers.append(result)
        if i != 0 and not config.print_full:
            print(f"  [{rer['qid']}] {result['answer'][:120]}...", flush=True)
    write_jsonl(os.path.join(run_dir, "05_answers.jsonl"), answers)
    timer.stage("4_5_prompt_and_generate")

    # ---- Report + diagnostics ----------------------------------------------
    banner("REPORT — HTML + three-failure-mode diagnostics")
    config_dump = {
        "n_docs": config.n_docs,
        "queries": len(queries),
        "embed_model": config.embed_model_id,
        "rerank_model": config.rerank_model_id,
        "llm": f"{config.llm_backend}:{config.llm_model_id}",
        "channel_top_k": config.channel_top_k,
        "fused_top_k": config.fused_top_k,
        "rerank_top_k": config.rerank_top_k,
        "device": device,
    }
    report_path = os.path.join(run_dir, "report.html")
    diagnostics = write_html_report(
        report_path, config_dump, timer.records,
        retrieval_results, rerank_results, answers,
    )
    with open(os.path.join(run_dir, "00_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config_dump, "timings": timer.records}, f, indent=2)
    timer.stage("report")

    show_artifact("failure-mode diagnostics summary", diagnostics["summary"], full=True)
    print("\nArtifacts:")
    for name in sorted(os.listdir(run_dir)):
        print(f"  {os.path.join(run_dir, name)}")
    print(f"\nOpen report: {report_path}")
    print(f"TOTAL elapsed: {sum(t for _, t in timer.records):.1f}s "
          f"(budget: 1800s on one GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
