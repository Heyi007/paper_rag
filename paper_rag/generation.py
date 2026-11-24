"""Stages 4-5 — prompt assembly and cited answer generation.

The prompt is fixed and deliberately simple: numbered passages, inline [n]
citations, no chain-of-thought, no rewriting. Three interchangeable LLM
backends are provided:

* hf     : local causal LM via transformers (default)
* openai : any OpenAI-compatible /chat/completions endpoint (env-configured)
* stub   : deterministic offline placeholder, for plumbing tests without a GPU
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from . import config as cfg
from .retrieval import passage_text
from .util import banner, show_artifact

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "provided paper abstracts.\n"
    "Rules:\n"
    "1. Support every factual claim with an inline citation using the passage "
    "number, e.g. [1] or [2].\n"
    "2. Only cite passage numbers that exist in the context ([1]-[5]); never "
    "invent citation numbers.\n"
    "3. Answer in 3-6 sentences.\n"
    "4. If the abstracts do not contain enough information, reply exactly: "
    "'I cannot answer from the provided sources.'"
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_prompt(query: str, top_docs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Assemble the user-turn context; references are 1-indexed in display."""
    blocks = []
    references = []
    for display_idx, doc in enumerate(top_docs, start=1):
        blocks.append(
            f"[{display_idx}] Title: {doc['title']}\n"
            f"Abstract: {doc['abstract']}\n"
            f"URL: {doc['url']}"
        )
        references.append(
            {
                "citation": display_idx,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": doc["url"],
            }
        )
    user_prompt = (
        "Passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}\n\nAnswer with citations:"
    )
    return user_prompt, references


def check_citations(answer: str, n_refs: int) -> dict[str, Any]:
    """Surface the 'citation mismatch' failure mode directly."""
    used = [int(x) for x in _CITATION_RE.findall(answer)]
    valid = sorted({n for n in used if 1 <= n <= n_refs})
    out_of_range = sorted({n for n in used if n < 1 or n > n_refs})
    uncited = sorted(set(range(1, n_refs + 1)) - set(valid))
    return {
        "citations_used": valid,
        "out_of_range_citations": out_of_range,
        "uncited_references": uncited,
        "ok": not out_of_range and bool(valid),
    }


class HFGenerator:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, user_prompt: str) -> str:
        torch = self.torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("HYTEST TEST ------------")
        print(messages)
        print("-------")
        print(text)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        print(inputs)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAIGenerator:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.base_url = os.environ["PAPER_RAG_LLM_BASE_URL"].rstrip("/")
        self.api_key = os.environ.get("PAPER_RAG_LLM_API_KEY", "EMPTY")

    def generate(self, user_prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": cfg.MAX_NEW_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


class StubGenerator:
    """Deterministic offline backend: always cites [1] and [2]."""

    def generate(self, user_prompt: str) -> str:
        return (
            "[STUB BACKEND] Based on the retrieved abstracts, the provided sources "
            "address this question directly in [1] and [2]. This is a deterministic "
            "placeholder answer used to validate the pipeline plumbing; switch "
            "--llm to hf or openai for real generation."
        )


def make_generator(config: cfg.PipelineConfig, device: str):
    if config.llm_backend == "hf":
        return HFGenerator(config.llm_model_id, device)
    if config.llm_backend == "openai":
        return OpenAIGenerator(config.llm_model_id)
    if config.llm_backend == "stub":
        return StubGenerator()
    raise ValueError(f"unknown llm backend: {config.llm_backend}")


def answer_query(
    config: cfg.PipelineConfig,
    generator,
    docs_by_id: dict[str, dict[str, Any]],
    reranked: dict[str, Any],
    prompt_dir: str,
    verbose: bool,
) -> dict[str, Any]:
    top_docs = [docs_by_id[item["doc_id"]] for item in reranked["top5"]]
    user_prompt, references = build_prompt(reranked["query"], top_docs)

    # Persist the exact assembled prompt (one file per query).
    prompt_path = os.path.join(prompt_dir, f"{reranked['qid']}_prompt.txt")
    os.makedirs(prompt_dir, exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}\n")

    answer = generator.generate(user_prompt)
    citation_report = check_citations(answer, len(references))
    result = {
        "qid": reranked["qid"],
        "query": reranked["query"],
        "prompt_path": prompt_path,
        "answer": answer,
        "references": references,
        "citation_check": citation_report,
        "top5_rerank": reranked["top5"],
    }
    if verbose:
        show_artifact(
            f"STAGE 4 prompt [{reranked['qid']}] (saved to {prompt_path})",
            {"system": SYSTEM_PROMPT, "user_head": user_prompt[:1200] + " ..."},
            config.print_full,
        )
        show_artifact(
            f"STAGE 5 answer [{reranked['qid']}]",
            {"answer": answer, "citation_check": citation_report, "references": references},
            config.print_full,
        )
    return result


def print_generation_banner() -> None:
    banner("STAGE 4-5/5  PROMPT + GENERATE — assemble cited prompt, LLM answers")
