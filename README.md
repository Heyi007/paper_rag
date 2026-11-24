# paper_rag

一个**最小但完整可运行**的检索增强生成（RAG）流水线：在 1 万篇 arXiv 论文摘要上，用「BM25 稀疏检索 + BGE 稠密检索」双路召回，经 RRF 融合与 Cross-Encoder 重排后，交给 LLM 生成带行内引用 `[n]` 的答案，并产出可点击的 HTML 报告与每一步的中间产物。

> **设计约束：不调参、不重写查询、无 Agent 循环。** 所有模型均为开箱即用的预训练模型，本仓库不含任何训练或超参搜索；CLI 参数只切换数据源 / 后端 / 规模，不改变算法选择。

## 流水线总览

```
10k arXiv 摘要
      │  Stage 1  构建/加载语料（远程 parquet 范围读，或本地 --from-file）
      ▼
BM25 top50  ┐
            ├─ RRF 融合 → fused top50        （Stage 2 双路检索）
BGE  top50  ┘
      ▼
Cross-Encoder 重排 → top5                     （Stage 3）
      ▼
编号段落 + 固定 prompt → LLM 生成带 [n] 引用答案 （Stage 4-5）
      ▼
HTML 报告 + 三类失败模式诊断 + 全量中间产物落盘
```

## 目录结构

```
paper_rag/
├── run.py                 # 入口：串起 5 个阶段，解析 CLI 参数
├── requirements.txt
├── queries.jsonl          # 默认查询集（单条，用于快速冒烟）
├── queries_all.jsonl      # 完整 30 条查询
├── paper_rag/
│   ├── config.py          # 固定配置：模型 ID、各阶段 top-k、RRF 常数等
│   ├── data.py            # Stage 1 语料构建（HF 远程 / 本地文件）
│   ├── retrieval.py       # Stage 2 BM25 + BGE 双路检索与 RRF 融合
│   ├── rerank.py          # Stage 3 Cross-Encoder 重排
│   ├── generation.py      # Stage 4-5 prompt 组装与三种 LLM 后端
│   ├── report.py          # HTML 报告与三类失败模式诊断
│   └── util.py            # JSONL IO、分词、计时、代理环境清理等工具
├── data/                  # 运行后生成（gitignore）：corpus.jsonl + 向量 .npy
└── artifacts/             # 运行后生成（gitignore）：每次 run 的全部产物
```

## 使用的模型（全部 off-the-shelf）

| 角色 | 模型 | 规模 | 说明 |
|---|---|---|---|
| 稠密向量 | `BAAI/bge-small-en-v1.5` | 33M | CLS pooling + L2 归一化，最长 512 token |
| 重排 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 联合编码 (query, passage)，输出单个相关性 logit |
| 生成 LLM | `Qwen/Qwen2.5-1.5B-Instruct`（默认） | 1.5B | 可用 `--llm-model` 替换 |

关键固定参数（见 [config.py](paper_rag/config.py)）：语料 `N_DOCS=10000`，每路召回 `top50`，RRF 融合后截断 `top50`，重排保留 `top5`，`RRF_K=60`，生成 `MAX_NEW_TOKENS=512`。

## 安装

需要 Python 3.10+，建议使用虚拟环境：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

首次运行会自动从 Hugging Face 下载上述三个模型到本地缓存。

## 快速开始

```bash
# 完整运行：单 GPU、1 万篇文档、30 条查询（约 30 分钟预算内）
python run.py

# 缩小规模，快速验证
python run.py --docs 1000 --query-limit 5

# 无 GPU 冒烟测试：stub 后端返回确定性占位答案，只验证管道衔接
python run.py --llm stub --device cpu --docs 200 --query-limit 3

# 使用本地语料文件（.jsonl/.parquet），完全离线
python run.py --from-file path/to/corpus.jsonl --llm stub

# 改用 OpenAI 兼容接口（需先设置环境变量，见下）
python run.py --llm openai --llm-model your-model-name
```

> 默认语料来自 Hugging Face 数据集 `librarian-bots/arxiv_abstracts`，通过 `HfFileSystem` + pyarrow 做 parquet 范围读取，不会整盘下载 280MB 分片，并优先保留 `cs.*` / `stat.ML` 类论文。要完全离线请通过 `--from-file` 提供本地语料。

### CLI 参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--docs` | `10000` | 语料规模 |
| `--query-limit` | `30` | 实际处理的查询条数 |
| `--queries-file` | `queries.jsonl` | 查询文件路径 |
| `--from-file` | 无 | 本地 `.jsonl`/`.parquet` 语料，替代远程 HF 源 |
| `--llm` | `hf` | 生成后端：`hf` / `openai` / `stub` |
| `--llm-model` | `Qwen/Qwen2.5-1.5B-Instruct` | 生成模型 ID |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--data-dir` | `./data` | 语料与向量缓存目录 |
| `--artifacts-dir` | `./artifacts` | 运行产物目录 |
| `--rebuild` | 关 | 忽略已缓存的语料 / 向量，强制重建 |
| `--print-full` | 关 | 控制台打印未截断的完整产物 |

## LLM 后端与环境变量

`generation.py` 提供三种可互换后端：

- **`hf`**（默认）：本地 transformers 因果模型，CUDA 下用 fp16、CPU 下用 fp32，贪心解码（`do_sample=False`）。
- **`openai`**：任意 OpenAI 兼容的 `/chat/completions` 端点，`temperature=0`。需配置：
  - `PAPER_RAG_LLM_BASE_URL`（**必填**）：服务基础地址，代码会拼接 `/chat/completions`
  - `PAPER_RAG_LLM_API_KEY`（可选，默认 `EMPTY`）
- **`stub`**：确定性离线占位回答（固定引用 `[1]`、`[2]`），用于无 GPU 时验证管道。

模型下载相关变量（代码内置了企业网络默认值，已导出的同名变量不会被覆盖）：

- `HF_ENDPOINT`：Hugging Face 镜像端点，可覆盖内置默认
- `HF_HUB_DISABLE_XET`：默认置 `1`

## 数据格式

查询文件为 JSONL，每行一条：

```json
{"qid": "q01", "question": "How does the Transformer architecture replace recurrence ... ?"}
```

本地语料（`--from-file`）每行需含 `title` 与 `abstract`，`arxiv_id`/`id`/`url`/`categories` 可选；标准化后的文档字段为 `doc_id, arxiv_id, title, abstract, url, categories`。

## 运行产物

每次运行在 `artifacts/run_<时间戳>/` 下生成：

```
artifacts/run_YYYYMMDD_HHMMSS/
├── 00_run_meta.json          # 配置快照与各阶段耗时
├── 02_retrieval.jsonl        # 每查询的 BM25/dense/RRF 三路结果
├── 03_rerank.jsonl           # fused top50 的重排得分与最终 top5
├── 04_prompts/qXX_prompt.txt # 每个查询实际拼装的 system+user prompt
├── 05_answers.jsonl          # 答案、引用校验、参考文档
├── report.html               # 可点击的 HTML 报告（引用可跳转）
└── report.diagnostics.json   # 三类失败模式诊断
```

语料与向量缓存位于 `data/`（`corpus.jsonl` 与 `embeddings.<model>.npy`），命中缓存时自动复用，`--rebuild` 可强制重建。

## 三类失败模式诊断

[report.py](paper_rag/report.py) 针对 RAG 的三种典型失效给出量化诊断：

1. **检索召不回（retrieval miss）**：统计 fused top50 中仅 BM25 命中、仅 dense 命中、两者共现的占比。
2. **排序错误（ranking error）**：衡量最终 top5 相对融合排名的位移，识别 Cross-Encoder 是否把低排名文档错误上提。
3. **引用对不上（citation mismatch）**：检查答案中的 `[n]` 是否越界、是否遗漏了已提供的参考文献；HTML 中越界引用会标红。

## 版本控制说明

`data/` 与 `artifacts/` 均在 `.gitignore` 中——语料、向量库和运行产物体积大且可由流水线重新生成，不纳入仓库；同理忽略虚拟环境、`.env`、`*.log` 与编辑器配置。Clone 后需自行运行以生成数据与产物，请勿提交本地密钥文件。
