# ✨ DocMagic

> Grounded, cited question-answering over your logistics, ERP and CRM documents.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-vector%20store-6C4BF6)
![License](https://img.shields.io/badge/License-MIT-green)

DocMagic is a retrieval-augmented document assistant for supply-chain and
operations teams. Upload SOPs, rate cards, contracts or inventory exports and
ask questions in plain language — every answer is grounded in your files and
cited to the exact page or sheet. A built-in logistics knowledge base covers
industry fundamentals, and an analysis mode turns spreadsheets into charts on
request.

## Demo

**Live app:** _deploy on Streamlit Community Cloud and drop the URL here_

<!-- Add a screenshot once deployed:  ![DocMagic](docs/screenshot.png) -->

## Features

- **Cited answers** — responses quote the exact source, e.g. `[rate_card.xlsx Rates]` or `[sop.pdf p.3]`, and the model answers only from retrieved context.
- **PDF and Excel ingestion** — page-level parsing for PDFs, sheet-level for spreadsheets, with automatic detection of header rows buried under title/metadata blocks.
- **Built-in domain knowledge** — a curated logistics/ERP/CRM knowledge base answers industry-fundamentals questions even before any file is uploaded.
- **Conversational analytics** — describe a chart in natural language ("pie of shipments by status") and DocMagic renders it; bar, line, area, scatter and pie are supported.
- **Live document preview** — uploaded PDFs render side-by-side with the chat.
- **Session isolation** — each visitor's uploaded documents are private to their session.
- **Provider-agnostic** — runs on any OpenAI-compatible endpoint; the chat model is selectable at runtime and the API key can be supplied by the operator or pasted per user.
- **Production hardening** — runtime secrets, upload limits, session-scoped storage, sanitised error handling with server-side logging, and no third-party telemetry.

## Architecture

```
Ingestion
  PDF / Excel ──▶ per-page / per-sheet documents ──▶ recursive text splitter
            ──▶ NVIDIA nemotron embeddings ──▶ Chroma (per-session collection)

Query
  Question ──▶ embed ──▶ retrieve top-k from { session documents + knowledge base }
           ──▶ LCEL chain (context + history ─▶ prompt ─▶ LLM)
           ──▶ streamed answer with citations
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Orchestration | LangChain (LCEL) |
| Vector store | Chroma (local, persistent) |
| Embeddings | NVIDIA NIM — `nemotron-3-embed-1b` |
| LLM | NVIDIA NIM — `nemotron-3-super` (default); any OpenAI-compatible model |
| Data & charts | pandas, Vega/Altair |

## Getting started

**Prerequisites:** Python 3.10+ and a free NVIDIA NIM API key ([build.nvidia.com](https://build.nvidia.com)).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # add your API key
streamlit run app.py
```

The knowledge base builds on demand from the sidebar (**Load industry
knowledge**), or ahead of time with `python kb_ingest.py` while the app is
stopped.

## Configuration

Set these in `.env`, or supply the key in the sidebar / Streamlit Secrets at runtime.

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for chat completions | _required_ |
| `LLM_BASE_URL` | OpenAI-compatible chat endpoint | `https://integrate.api.nvidia.com/v1` |
| `LLM_MODEL` | Chat model id | `nvidia/nemotron-3-super-120b-a12b` |
| `EMBED_API_KEY` | API key for embeddings | falls back to `LLM_API_KEY` |
| `EMBED_BASE_URL` | Embeddings endpoint | `https://integrate.api.nvidia.com/v1` |
| `EMBED_MODEL` | Embedding model id | `nvidia/nemotron-3-embed-1b` |

## Deployment

Deploy on [Streamlit Community Cloud](https://share.streamlit.io): point it at
`app.py`, and either set `LLM_API_KEY` in the app's **Secrets** or let each
visitor paste their own key. Storage on the free tier is ephemeral — rebuild
the knowledge base after a restart with the sidebar button.

## Project structure

```
app.py                  # Streamlit app: UI, RAG pipeline, analytics
kb_ingest.py            # Builds the logistics knowledge base
test_rag.py             # Unit tests: chunking, retrieval, Excel parsing
.streamlit/config.toml  # Theme and server configuration
finetune/               # Domain fine-tuning kit (Llama 3.2 3B + QLoRA)
```

## Testing

```bash
python test_rag.py
```

## Roadmap

- Reranking for higher retrieval precision
- Hybrid keyword + vector search
- Domain-tuned model (see [`finetune/`](finetune/))

## License

Released under the [MIT License](LICENSE).
