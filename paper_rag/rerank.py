"""Stage 3 — cross-encoder reranking of the fused top-50 -> top-5.

The cross-encoder reads (query, passage) pairs jointly and emits one relevance
logit per pair. It is slower per pair than bi-encoder retrieval but is only run
on the 50 fused candidates, i.e. 30 x 50 = 1,500 pairs for the full run.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import config as cfg
from .retrieval import passage_text
from .util import banner, show_artifact


class CrossEncoderReranker:
    def __init__(self, model_id: str, device: str, batch_size: int, max_len: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        )

    def score_pairs(self, query: str, passages: list[str]) -> np.ndarray:
        torch = self.torch
        scores = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            scores.append(logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(scores, axis=0)


def rerank(
    config: cfg.PipelineConfig,
    reranker: CrossEncoderReranker,
    docs_by_id: dict[str, dict[str, Any]],
    retrieval: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    candidates = retrieval["fused_top50"]
    passages = [passage_text(docs_by_id[c["doc_id"]]) for c in candidates]
    scores = reranker.score_pairs(retrieval["query"], passages)

    scored = []
    for cand, score in zip(candidates, scores):
        scored.append(
            {
                "doc_id": cand["doc_id"],
                "retrieval_rank": cand["rank"],
                "channels": cand["channels"],
                "rerank_score": round(float(score), 4),
            }
        )
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rerank_rank"] = rank
    top5 = scored[: config.rerank_top_k]

    result = {
        "qid": retrieval["qid"],
        "query": retrieval["query"],
        "candidates_scored": scored,
        "top5": top5,
    }
    if verbose:
        show_artifact(
            f"STAGE 3 rerank [{retrieval['qid']}] top5 after cross-encoder",
            top5,
            config.print_full,
        )
    return result


def print_rerank_banner() -> None:
    banner("STAGE 3/5  RERANK — cross-encoder scores fused top50 -> top5")
