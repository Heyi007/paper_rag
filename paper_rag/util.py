"""Shared helpers: JSONL IO, text normalization, timing and artifact printing."""
from __future__ import annotations

import json
import os
import re
import time
import warnings
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?:/\d+)?$")
_PROXY_VALUE_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
_NO_PROXY_VARS = ("NO_PROXY", "no_proxy")
_PROXY_SANITIZED = False


def _proxy_pattern_key(entry: str, httpx_module) -> str:
    """Mirror httpx's get_environment_proxies mount-key construction exactly."""
    if "://" in entry:
        return entry
    try:
        from httpx._utils import is_ipv4_hostname, is_ipv6_hostname

        if is_ipv4_hostname(entry):
            return f"all://{entry}"
        if is_ipv6_hostname(entry):     # bare IPv6 literal, e.g. ::1
            return f"all://[{entry}]"
    except ImportError:                  # pragma: no cover - private API moved
        if _IPV4_RE.match(entry):
            return f"all://{entry}"
    return f"all://*{entry}"            # domains, localhost, host:port styles


def _pattern_parses(httpx_module, key: str) -> bool:
    try:
        from httpx._utils import URLPattern

        URLPattern(key)
        return True
    except ImportError:                  # pragma: no cover - private API moved
        try:
            httpx_module.URL(key)
            return True
        except httpx_module.InvalidURL:
            return False
    except httpx_module.InvalidURL:
        return False


def sanitize_proxy_env() -> list[str]:
    """Drop malformed proxy / no-proxy env entries before httpx reads them.

    Some platform-bootstrapped environments ship values such as a NO_PROXY
    entry ending in ':' or a proxy URL with a doubled port separator. httpx
    parses every proxy mount at Client construction and raises
    ``httpx.InvalidURL: Invalid port: ':'``, which crashes huggingface_hub
    before any request is sent. Entries that cannot be parsed the way httpx
    parses them are removed here; valid proxies and bypass entries are kept.
    Idempotent: only the first call performs work.
    """
    global _PROXY_SANITIZED
    if _PROXY_SANITIZED:
        return []
    _PROXY_SANITIZED = True
    dropped: list[str] = []
    try:
        import httpx
    except ImportError:
        return dropped

    for name in _PROXY_VALUE_VARS:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        candidate = value if "://" in value else f"http://{value}"
        try:
            httpx.URL(candidate)
        except httpx.InvalidURL:
            dropped.append(f"{name}={value!r}")
            os.environ.pop(name, None)

    for name in _NO_PROXY_VARS:
        raw = os.environ.get(name)
        if not raw:
            continue
        kept: list[str] = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if _pattern_parses(httpx, _proxy_pattern_key(entry, httpx)):
                kept.append(entry)
            else:
                dropped.append(f"{name} entry {entry!r}")
        os.environ[name] = ",".join(kept)

    if dropped:
        warnings.warn(
            "removed malformed proxy environment entries so httpx can start: "
            + "; ".join(dropped),
            stacklevel=2,
        )
        print(f"[env] dropped malformed proxy entries: {'; '.join(dropped)}", flush=True)
    return dropped


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs (arXiv abstracts contain hard newlines)."""
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    """Lowercase regex tokenizer: no stopword removal, no stemming."""
    return _TOKEN_RE.findall(text.lower())


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Timer:
    """Stage timer that also keeps a cumulative wall-clock log."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.records: list[tuple[str, float]] = []

    def stage(self, name: str) -> float:
        now = time.time()
        elapsed = now - self.t0
        self.records.append((name, elapsed))
        self.t0 = now
        return elapsed


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def show_artifact(title: str, payload: Any, full: bool, max_chars: int = 1600) -> None:
    """Print an intermediate artifact.

    Full content is always persisted to disk by the caller; the console view is
    truncated unless --print-full is passed, so a 30-query run stays readable.
    """
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(f"\n--- {title} ---", flush=True)
    if full or len(rendered) <= max_chars:
        print(rendered, flush=True)
    else:
        print(rendered[:max_chars] + f"\n... [truncated {len(rendered) - max_chars} chars, full artifact saved to disk]", flush=True)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
