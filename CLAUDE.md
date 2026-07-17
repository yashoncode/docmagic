# DocMagic — context for Claude

Yashwanth's first AI-career portfolio project (works at Stockarea — logistics/
warehousing domain). Goal: resume-worthy, everything free/open-source, built
from scratch without LangChain so every part is explainable in interviews.

Branch scheme: `master` = from-scratch pipeline + Gradio; `langchain` (this
branch) = LangChain (loaders/splitter/LCEL) + Streamlit. Same behavior, same
.env, comparable line by line.

Stack (this branch): Streamlit + LangChain + Chroma (NIM-hosted nemotron
embeddings) + any OpenAI-compatible LLM API (NVIDIA NIM free tier by default).
Run: `streamlit run app.py`. Test: `python test_rag.py`.

Roadmap lives in README.md; phase 2 (QLoRA fine-tune on Colab) in finetune/.
Keep it simple — no frameworks or abstractions unless asked.

Never add AI attribution to commits or PRs (no Co-Authored-By, no "Generated
with"). All commits authored solely by Yashwanth D.
