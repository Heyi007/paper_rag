"""HTML report + diagnostics for the three canonical RAG failure modes:

1. retrieval miss  ("召不回")  — one channel keeps finding things the other misses;
   measured by BM25-only / dense-only / shared membership of the fused top-50.
2. ranking error   ("排序错")  — cross-encoder promotes documents that retrieval
   ranked low; measured by displacement of the final top-5 vs fused rank.
3. citation mismatch ("引用对不上") — answers cite numbers that do not exist, or
   leave provided references uncited.
"""
from __future__ import annotations

import html
import json
import os
import re
from typing import Any

_CITE_RE = re.compile(r"\[(\d+)\]")


def _link_citations(answer: str, qid: str, n_refs: int) -> str:
    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        safe = html.escape(m.group(0))
        if 1 <= n <= n_refs:
            return f'<a class="cite" href="#{qid}-ref-{n}">{safe}</a>'
        return f'<span class="cite bad" title="citation {n} does not exist">{safe}</span>'

    return _CITE_RE.sub(repl, html.escape(answer))


def build_diagnostics(
    retrieval_results: list[dict[str, Any]],
    rerank_results: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    per_query = []
    for retr, rer, ans in zip(retrieval_results, rerank_results, answers):
        fused = {c["doc_id"]: c for c in retr["fused_top50"]}
        bm25_ids = {h["doc_id"] for h in retr["bm25_top50"]}
        dense_ids = {h["doc_id"] for h in retr["dense_top50"]}
        both = bm25_ids & dense_ids
        top5_ranks = [fused[item["doc_id"]]["rank"] for item in rer["top5"]]
        per_query.append({
            "qid": ans["qid"],
            "channels": {
                "bm25_only": len(bm25_ids - dense_ids),
                "dense_only": len(dense_ids - bm25_ids),
                "in_both": len(both),
            },
            "ranking": {
                "top5_outside_fused_top10": sum(1 for r in top5_ranks if r > 10),
                "max_retrieval_rank_in_top5": max(top5_ranks),
            },
            "citation": ans["citation_check"],
        })

    n = len(per_query)
    summary = {
        "retrieval_miss_signal": {
            "avg_bm25_only_docs": round(sum(q["channels"]["bm25_only"] for q in per_query) / n, 1),
            "avg_dense_only_docs": round(sum(q["channels"]["dense_only"] for q in per_query) / n, 1),
            "avg_shared_docs": round(sum(q["channels"]["in_both"] for q in per_query) / n, 1),
        },
        "ranking_error_signal": {
            "queries_with_top5_promoted_from_below_10": sum(
                q["ranking"]["top5_outside_fused_top10"] > 0 for q in per_query
            ),
            "worst_observed_retrieval_rank_in_top5": max(
                q["ranking"]["max_retrieval_rank_in_top5"] for q in per_query
            ),
        },
        "citation_mismatch_signal": {
            "queries_with_out_of_range_citations": sum(
                1 for q in per_query if q["citation"]["out_of_range_citations"]
            ),
            "queries_without_any_valid_citation": sum(
                not q["citation"]["ok"] for q in per_query
            ),
            "total_uncited_references": sum(
                len(q["citation"]["uncited_references"]) for q in per_query
            ),
        },
    }
    return {"summary": summary, "per_query": per_query}


def _render_card(ans: dict[str, Any], rer: dict[str, Any], retr: dict[str, Any]) -> str:
    qid = ans["qid"]
    n_refs = len(ans["references"])
    refs_html = []
    score_by_id = {item["doc_id"]: item for item in rer["candidates_scored"]}
    for ref in ans["references"]:
        n = ref["citation"]
        extra = score_by_id.get(ref["doc_id"], {})
        refs_html.append(
            f'<div class="ref" id="{qid}-ref-{n}">'
            f'<span class="refnum">[{n}]</span> '
            f'<a href="{html.escape(ref["url"])}" target="_blank" rel="noopener">'
            f'{html.escape(ref["title"])}</a>'
            f'<div class="meta">{html.escape(ref["doc_id"])} · '
            f'rerank score {extra.get("rerank_score", "?")} · '
            f'retrieval rank {extra.get("retrieval_rank", "?")} · '
            f'channels {",".join(extra.get("channels", []))}</div></div>'
        )
    cc = ans["citation_check"]
    badge = (
        '<span class="badge ok">citations ok</span>'
        if cc["ok"]
        else f'<span class="badge bad">citation problem: {html.escape(json.dumps(cc))}</span>'
    )
    channels = {c["doc_id"]: c["channels"] for c in retr["fused_top50"][:5]}
    head = ", ".join(
        f'{d["doc_id"]}({"/".join(channels.get(d["doc_id"], []))})' for d in rer["top5"]
    )
    return f"""
    <section class="card">
      <h2>{html.escape(qid)} · {html.escape(ans["query"])}</h2>
      <div class="answer">{_link_citations(ans["answer"], qid, n_refs)}</div>
      {badge}
      <details><summary>retrieval / rerank trace (top5)</summary>
        <p class="meta">fused top5 doc ids and contributing channels: {html.escape(head)}</p>
        <table><tr><th>rerank rank</th><th>doc</th><th>fused rank</th><th>channels</th><th>rerank score</th></tr>
        {''.join(
            f'<tr><td>{item["rerank_rank"]}</td><td>{item["doc_id"]}</td>'
            f'<td>{item["retrieval_rank"]}</td><td>{",".join(item["channels"])}</td>'
            f'<td>{item["rerank_score"]}</td></tr>'
            for item in rer["candidates_scored"][:5]
        )}
        </table>
      </details>
      <h3>References (click to open arXiv)</h3>
      {''.join(refs_html)}
    </section>"""


def write_html_report(
    out_path: str,
    config_dump: dict[str, Any],
    timings: list[tuple[str, float]],
    retrieval_results: list[dict[str, Any]],
    rerank_results: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = build_diagnostics(retrieval_results, rerank_results, answers)
    cards = [
        _render_card(ans, rer, retr)
        for ans, rer, retr in zip(answers, rerank_results, retrieval_results)
    ]
    timing_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{elapsed:.1f}s</td></tr>" for name, elapsed in timings
    )
    diag_summary = diagnostics["summary"]
    diag_html = f"""
    <section class="card diag">
      <h2>Failure-mode diagnostics</h2>
      <h3>1. Retrieval miss (召不回) — channel disagreement in fused top-50</h3>
      <pre>{html.escape(json.dumps(diag_summary["retrieval_miss_signal"], indent=2))}</pre>
      <h3>2. Ranking error (排序错) — cross-encoder vs retrieval rank</h3>
      <pre>{html.escape(json.dumps(diag_summary["ranking_error_signal"], indent=2))}</pre>
      <h3>3. Citation mismatch (引用对不上)</h3>
      <pre>{html.escape(json.dumps(diag_summary["citation_mismatch_signal"], indent=2))}</pre>
    </section>"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>paper_rag report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: #f7f7f9; color: #222; }}
h1 {{ margin-bottom: 4px; }}
.card {{ background: #fff; border: 1px solid #e3e3e8; border-radius: 10px; padding: 18px 22px; margin: 18px 0; }}
.answer {{ white-space: pre-wrap; line-height: 1.55; font-size: 15px; }}
.cite {{ text-decoration: none; color: #0b5fff; font-weight: 600; }}
.cite.bad {{ color: #c0392b; border-bottom: 1px dashed #c0392b; }}
.ref {{ margin: 8px 0; padding: 8px 10px; background: #fafafe; border-radius: 6px; }}
.refnum {{ font-weight: 700; color: #555; }}
.meta {{ color: #777; font-size: 12px; margin-top: 4px; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; margin: 8px 0; }}
.badge.ok {{ background: #e6f6ec; color: #1a7f37; }}
.badge.bad {{ background: #fdecea; color: #c0392b; }}
table {{ border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
pre {{ background: #f4f4f7; padding: 10px; border-radius: 6px; font-size: 12px; overflow-x: auto; }}
details {{ margin: 10px 0; }}
</style></head><body>
<h1>paper_rag — BM25 + BGE + cross-encoder RAG run</h1>
<p class="meta">{html.escape(json.dumps(config_dump, ensure_ascii=False))}</p>
<details open><summary>Stage timings</summary><table>{timing_rows}</table></details>
{diag_html}
{''.join(cards)}
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    diag_path = out_path.replace(".html", ".diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    return diagnostics
