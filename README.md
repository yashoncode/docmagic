# LogiChat 📦 — chat with your logistics / ERP / CRM documents

Ask questions about SOPs, rate cards, contracts and warehouse manuals in plain
language and get answers **cited to the exact file and page**.

Built from scratch (no LangChain) to understand every moving part of a RAG
pipeline: PDF parsing -> chunking -> local embeddings -> vector search ->
grounded, streamed LLM answers.

## Architecture

```
PDFs -> pypdf -> word-window chunks (250 words, 50 overlap)
     -> Chroma (local, persistent; MiniLM ONNX embeddings — free, no GPU)
     -> top-5 retrieval per question
     -> Llama 3.1 8B via NVIDIA NIM free API (any OpenAI-compatible endpoint works)
     -> streamed answer with [file.pdf p.N] citations
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then paste your free key from build.nvidia.com
python app.py                   # opens the Gradio UI
```

Upload PDFs, click **Ingest**, ask away.

## Tests

```bash
python test_rag.py
```

## Roadmap

- [x] RAG over domain PDFs with page-level citations
- [ ] Fine-tuned domain model (Llama 3.2 3B + QLoRA on logistics Q&A) — see [finetune/](finetune/)
- [ ] Eval harness: base model vs fine-tuned, retrieval hit-rate
- [ ] Deploy demo on Hugging Face Spaces
