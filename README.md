# DocMagic ✨ — chat with your Docs (supports PDFs and Excel)

Ask questions about SOPs, rate cards, contracts and warehouse manuals in plain
language and get answers **cited to the exact file and page**.

**This branch (`langchain`)** is the same app rebuilt on LangChain with a
Streamlit frontend, so the implementations can be compared line by line.

## Branches

| branch | RAG pipeline | UI |
|--------|--------------|-----|
| `master` | from scratch (no framework) | Gradio |
| `langchain` | LangChain (loaders, splitter, LCEL chain) | Streamlit |

## Architecture

```
PDFs (PyPDFLoader) / Excel (custom loader) -> Documents per page/sheet
     -> RecursiveCharacterTextSplitter (1500 chars, 300 overlap)
     -> Chroma vector store (local, persistent; NVIDIAEmbeddings nemotron-3-embed-1b)
     -> LCEL chain: retriever | prompt | LLM  (NVIDIA NIM free API)
     -> streamed answer with [file.pdf p.N] citations
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then paste your free key from build.nvidia.com
streamlit run app.py
```

Upload PDFs or Excel files (.xlsx), click **Ingest**, ask away.

## Tests

```bash
python test_rag.py
```

## Roadmap

- [x] RAG over domain PDFs with page-level citations
- [ ] Fine-tuned domain model (Llama 3.2 3B + QLoRA on logistics Q&A) — see [finetune/](finetune/)
- [ ] Eval harness: base model vs fine-tuned, retrieval hit-rate
- [ ] Deploy demo on Hugging Face Spaces
