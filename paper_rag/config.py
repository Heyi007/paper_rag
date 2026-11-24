"""Fixed pipeline configuration.

The exercise explicitly forbids tuning, so every value below is a documented
default rather than a searchable hyper-parameter. CLI flags only switch data
sources / backends / scale, they do not change the algorithmic choices.
"""
from __future__ import annotations

import dataclasses
import os

# --- Off-the-shelf models (all fetched through the local HF cache) ---------
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"          # 33M, CLS pooling, 512 tok
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22M, 1 label
DEFAULT_LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"      # small instruct model

# --- Pipeline shape ----------------------------------------------------------
N_DOCS = 10_000           # corpus size: 10k paper abstracts
CHANNEL_TOP_K = 50        # each retriever returns 50
FUSED_TOP_K = 50          # RRF union is cut to 50 before reranking
RERANK_TOP_K = 5          # cross-encoder keeps 5
RRF_K = 60                # standard reciprocal-rank-fusion constant
EMBED_MAX_LEN = 512
RERANK_MAX_LEN = 512
EMBED_BATCH = 64
RERANK_BATCH = 64
MAX_NEW_TOKENS = 512

# --- Corpus source (librarian-bots/arxiv_abstracts big shards) --------------
HF_DATASET_REPO = "librarian-bots/arxiv_abstracts"
# The repo nominally declares 7 shards but only 6 are published; use exact names.
HF_SHARD_FILES = [
    "data/train-00000-of-00007-93dc4f049a0266e7.parquet",
    "data/train-00001-of-00007-905f8c5c61fc5538.parquet",
    "data/train-00002-of-00007-5a8d11291766cd76.parquet",
    "data/train-00003-of-00007-ce851835c6a439e9.parquet",
    "data/train-00004-of-00007-d139fd477db91fd5.parquet",
    "data/train-00005-of-00007-4b68f88c5f8501f5.parquet",
]
CS_PREFIXES = ("cs.", "stat.ML")
MAX_ROW_GROUPS = 100     # safety bound on ranged parquet reads

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_QUERIES_FILE = os.path.join(PROJECT_ROOT, "queries.jsonl")


@dataclasses.dataclass
class PipelineConfig:
    n_docs: int = N_DOCS
    channel_top_k: int = CHANNEL_TOP_K
    fused_top_k: int = FUSED_TOP_K
    rerank_top_k: int = RERANK_TOP_K
    rrf_k: int = RRF_K
    embed_model_id: str = EMBED_MODEL_ID
    rerank_model_id: str = RERANK_MODEL_ID
    llm_model_id: str = DEFAULT_LLM_ID
    llm_backend: str = "hf"          # hf | openai | stub
    device: str = "auto"             # auto | cuda | cpu
    data_dir: str = DEFAULT_DATA_DIR
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
    queries_file: str = DEFAULT_QUERIES_FILE
    from_file: str | None = None     # local .jsonl/.parquet corpus override
    query_limit: int = 30
    print_full: bool = False
    rebuild: bool = False

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.data_dir, "corpus.jsonl")

    @property
    def embeddings_path(self) -> str:
        slug = self.embed_model_id.replace("/", "_").replace(".", "_")
        return os.path.join(self.data_dir, f"embeddings.{slug}.npy")
