"""LogiChat — chat with your logistics / ERP / CRM documents.

RAG pipeline: PDF -> chunks -> Chroma (local, free) -> LLM via any
OpenAI-compatible API (NVIDIA NIM free tier by default).
"""

import os

import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "missing-key-see-.env.example"
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

CHUNK_WORDS = 250
CHUNK_OVERLAP = 50
TOP_K = 5

SYSTEM_PROMPT = (
    "You are LogiChat, an assistant for logistics, warehousing, ERP and CRM "
    "documents (SOPs, rate cards, contracts, manuals). Answer ONLY from the "
    "provided context. Cite the source file and page like [file.pdf p.3]. "
    "If the context does not contain the answer, say so plainly."
)

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
# Chroma's default embedding = all-MiniLM-L6-v2 via ONNX: local, free, no torch.
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("docs")


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def ingest(files: list[str]) -> str:
    """Extract, chunk and index the uploaded PDFs."""
    if not files:
        return "Upload at least one PDF."
    total = 0
    for path in files:
        name = os.path.basename(path)
        reader = PdfReader(path)
        for page_no, page in enumerate(reader.pages, start=1):
            for i, chunk in enumerate(chunk_text(page.extract_text() or "")):
                collection.add(
                    ids=[f"{name}-p{page_no}-c{i}"],
                    documents=[chunk],
                    metadatas=[{"source": name, "page": page_no}],
                )
                total += 1
    return f"Indexed {total} chunks. Collection now holds {collection.count()}."


def retrieve(question: str) -> list[tuple[str, dict]]:
    res = collection.query(query_texts=[question], n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(message: str, history: list[dict]):
    if collection.count() == 0:
        yield "No documents indexed yet — upload PDFs and click Ingest first."
        return
    hits = retrieve(message)
    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{doc}" for doc, m in hits
    )
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        # keep only the keys the OpenAI API accepts — gradio adds extras
        + [{"role": m["role"], "content": m["content"]} for m in history]
        + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}]
    )
    stream = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.2, stream=True
    )
    text = ""
    for event in stream:
        text += event.choices[0].delta.content or ""
        yield text


with gr.Blocks(title="LogiChat") as demo:
    gr.Markdown("# LogiChat 📦\nChat with your logistics / ERP / CRM documents.")
    with gr.Row():
        files = gr.File(file_count="multiple", file_types=[".pdf"], label="PDFs")
        with gr.Column():
            ingest_btn = gr.Button("Ingest", variant="primary")
            status = gr.Textbox(label="Index status", interactive=False)
    ingest_btn.click(ingest, inputs=files, outputs=status)
    gr.ChatInterface(answer)

if __name__ == "__main__":
    demo.launch()
