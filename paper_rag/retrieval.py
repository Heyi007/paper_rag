"""Stage 2 — dual-path retrieval: BM25 (lexical) + BGE (dense), fused by RRF.

No query rewriting, no HyDE, no learned fusion: each channel returns its own
top-50 and the two ranked lists are merged with reciprocal rank fusion.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from . import config as cfg
from .util import banner, show_artifact, tokenize


def passage_text(doc: dict[str, Any]) -> str:
    return f"{doc['title']}\n{doc['abstract']}"


def _topk(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]


class DenseEncoder:
    """bge-small-en-v1.5 with CLS pooling + L2 normalization (model card)."""

    def __init__(self, model_id: str, device: str, batch_size: int, max_len: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            cls = hidden[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class Retriever:
    def __init__(self, config: cfg.PipelineConfig, docs: list[dict[str, Any]], device: str):
        self.config = config
        self.docs = docs
        self.device = device
        self.passages = [passage_text(d) for d in docs]
        self._bm25 = self._build_bm25()
        self.doc_embeddings = self._build_dense()

    def _build_bm25(self) -> BM25Okapi:
        tokenized = [tokenize(p) for p in self.passages]
        return BM25Okapi(tokenized)

    def _build_dense(self) -> np.ndarray:
        self.encoder = DenseEncoder(
            self.config.embed_model_id,
            self.device,
            cfg.EMBED_BATCH,
            cfg.EMBED_MAX_LEN,
        )
        path = self.config.embeddings_path
        if not self.config.rebuild and os.path.exists(path):
            emb = np.load(path)
            if emb.shape[0] == len(self.docs):
                print(f"dense: loaded cached embeddings {emb.shape} from {path}", flush=True)
                return emb
        emb = self.encoder.encode(self.passages)
        np.save(path, emb)
        print(f"dense: encoded {emb.shape} -> cached at {path}", flush=True)
        return emb

    def _bm25_search(self, query: str) -> list[dict[str, Any]]:
        scores = self._bm25.get_scores(tokenize(query))
        return [
            {"doc_id": self.docs[i]["doc_id"], "rank": r + 1, "score": round(s, 4)}
            for r, (i, s) in enumerate(_topk(np.asarray(scores), self.config.channel_top_k))
        ]

    def _dense_search(self, query: str) -> list[dict[str, Any]]:
        qv = self.encoder.encode_one(query)
        sims = self.doc_embeddings @ qv
        return [
            {"doc_id": self.docs[i]["doc_id"], "rank": r + 1, "score": round(s, 4)}
            for r, (i, s) in enumerate(_topk(sims, self.config.channel_top_k))
        ]

    def _rrf_fuse(
        self, bm25_hits: list[dict[str, Any]], dense_hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rrf: dict[str, dict[str, Any]] = {}
        for channel, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
            for hit in hits:
                slot = rrf.setdefault(
                    hit["doc_id"], {"doc_id": hit["doc_id"], "rrf_score": 0.0, "channels": []}
                )
                slot["rrf_score"] += 1.0 / (self.config.rrf_k + hit["rank"])
                slot["channels"].append(channel)
        fused = sorted(rrf.values(), key=lambda x: x["rrf_score"], reverse=True)
        fused = fused[: self.config.fused_top_k]
        for rank, slot in enumerate(fused, start=1):
            slot["rank"] = rank
            slot["rrf_score"] = round(slot["rrf_score"], 6)
        return fused

    def search(self, qid: str, query: str, verbose: bool) -> dict[str, Any]:
        bm25_hits = self._bm25_search(query)
        dense_hits = self._dense_search(query)
        fused = self._rrf_fuse(bm25_hits, dense_hits)
        result = {
            "qid": qid,
            "query": query,
            "bm25_top50": bm25_hits,
            "dense_top50": dense_hits,
            "fused_top50": fused,
        }
        if verbose:
            show_artifact(
                f"STAGE 2 retrieval [{qid}] BM25 top5 / dense top5 / RRF top5",
                {
                    "bm25_top5": bm25_hits[:5],
                    "dense_top5": dense_hits[:5],
                    "fused_top5": fused[:5],
                },
                self.config.print_full,
            )
        return result


def print_retrieval_banner() -> None:
    banner("STAGE 2/5  RETRIEVAL — BM25 + BGE dual path, RRF fusion -> top 50")
