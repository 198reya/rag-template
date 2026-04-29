# simple-rag-template

A production-ready simple RAG template. Drop documents in, ask questions, get grounded answers with citations.

## Stack

| Layer         | Choice                                  |
|---------------|-----------------------------------------|
| LLM           | OpenRouter (`openai/gpt-oss-120b:free`) |
| Embeddings    | `sentence-transformers` (runs locally)  |
| Dense search  | FAISS                                   |
| Sparse search | BM25 (`rank-bm25`)                      |
| Fusion        | Reciprocal Rank Fusion (RRF)            |

## Project structure

```
rag-template/
├── data/                  ← drop your .txt / .pdf / .md files here
├── src/
│   ├── ingest.py          ← file loading + recursive text splitter
│   ├── vector_store.py    ← FAISS + BM25 + RRF hybrid search
│   ├── generation.py      ← grounded prompt template + LLM call + citations
│   └── embedder.py        ← sentence-transformers wrapper
├── pipeline.py            ← orchestrates ingest + query
├── main.py                ← CLI entry point
├── config.yaml            ← all settings in one place
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# open .env and paste your OPENROUTER_API_KEY
```

## Usage

1. Copy `.txt`, `.pdf`, or `.md` files into `data/`.
2. Run:

```bash
python main.py
```

The pipeline indexes your documents on startup, then opens an interactive prompt.
Each answer shows which files and page numbers it drew from.

## Configuration (`config.yaml`)

| Key                      | Default                    | Effect                                    |
|--------------------------|----------------------------|-------------------------------------------|
| `llm.model`              | `openai/gpt-oss-120b:free` | Any OpenRouter model string               |
| `llm.temperature`        | `0.1`                      | Lower = more deterministic                |
| `chunking.chunk_size`    | `800`                      | Max characters per chunk                  |
| `chunking.chunk_overlap` | `150`                      | Characters shared between adjacent chunks |
| `retrieval.top_k`        | `5`                        | Chunks fed to the LLM                     |
| `retrieval.hybrid_alpha` | `0.7`                      | Reserved for weighted fusion tuning       |

## How hybrid search works

Each query runs two retrievers in parallel:

1. **Dense (FAISS)** — finds semantically similar chunks via cosine-style L2 distance on embeddings.
2. **Sparse (BM25)** — finds chunks containing the exact query terms.

The two ranked lists are merged with **Reciprocal Rank Fusion (RRF)**, which rewards chunks that rank highly in both lists. This catches cases where semantic search misses an exact keyword match and vice versa.

## Swapping the model

Edit `config.yaml`:

```yaml
llm:
  model: anthropic/claude-3.5-sonnet
```

Any model on [openrouter.ai/models](https://openrouter.ai/models) works.
