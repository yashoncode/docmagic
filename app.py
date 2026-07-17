"""DocMagic — chat with your Docs (PDFs and Excel).

RAG pipeline: PDF/Excel -> chunks -> Chroma (local, free) -> LLM via any
OpenAI-compatible API (NVIDIA NIM free tier by default).
"""

import logging
import os

import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import load_workbook
from pypdf import PdfReader

load_dotenv()

# root stays at WARNING so httpx/gradio chatter is hidden; only our logs show
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("docmagic")
log.setLevel(logging.INFO)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "missing-key-see-.env.example"
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or LLM_API_KEY
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

CHUNK_WORDS = 250
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_FILES = 2
SUMMARY_WORDS = 1500  # per file; keeps the summary prompt small and cheap

SYSTEM_PROMPT = (
    "You are DocMagic, an assistant for logistics, warehousing, ERP and CRM "
    "documents (SOPs, rate cards, contracts, manuals). Answer ONLY from the "
    "provided context. Cite the source like [file.pdf p.3] or [file.xlsx Rates] "
    "(sheet name for spreadsheets). "
    "If the context does not contain the answer, say so plainly."
)

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
embed_client = OpenAI(base_url=LLM_BASE_URL, api_key=EMBED_API_KEY)
# embeddings come from the NIM API (see embed()); Chroma only stores/searches them
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("docs")


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed via NIM. input_type: 'passage' for docs, 'query' for questions."""
    res = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [d.embedding for d in res.data]


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


def extract_units(path: str):
    """Yield (location_label, text) per PDF page or Excel sheet."""
    if path.lower().endswith((".xlsx", ".xlsm")):
        wb = load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = (
                " | ".join(str(c) for c in row if c is not None)
                for row in ws.iter_rows(values_only=True)
            )
            yield ws.title, "\n".join(r for r in rows if r)
        wb.close()
    else:
        reader = PdfReader(path)
        for page_no, page in enumerate(reader.pages, start=1):
            yield f"p.{page_no}", page.extract_text() or ""


def ingest(files: list[str]) -> str:
    """Extract, chunk and index the uploaded documents."""
    if not files:
        return "Upload at least one PDF or Excel file."
    total = 0
    for path in files:
        name = os.path.basename(path)
        ids, docs, metas = [], [], []
        for loc, text in extract_units(path):
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{name}-{loc}-c{i}")
                docs.append(chunk)
                metas.append({"source": name, "loc": loc})
        for s in range(0, len(ids), 50):  # ponytail: 50/batch, NIM rejects huge ones
            b = slice(s, s + 50)
            collection.add(
                ids=ids[b],
                documents=docs[b],
                metadatas=metas[b],
                embeddings=embed(docs[b], "passage"),
            )
        total += len(ids)
    return f"Indexed {total} chunks. Collection now holds {collection.count()}."


def stream_llm(messages: list[dict]):
    """Stream a chat completion, yielding the accumulated text."""
    stream = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.2, stream=True
    )
    text = ""
    for event in stream:
        text += event.choices[0].delta.content or ""
        yield text
    log.info("LLM response <- %d chars: %s", len(text), text)


def analyse(files: list[str]):
    """Index the uploaded files (fresh collection) and stream a summary."""
    global collection
    if not files:
        yield "Upload at least one file first."
        return
    if len(files) > MAX_FILES:
        yield f"Please upload at most {MAX_FILES} files at a time."
        return
    yield "Reading and indexing…"
    # each Submit & Analyse starts fresh so chat only covers the current files
    db.delete_collection("docs")
    collection = db.get_or_create_collection("docs")
    ingest(files)
    excerpts = []
    for path in files:
        words = []
        for _, text in extract_units(path):
            words += text.split()
            if len(words) >= SUMMARY_WORDS:
                break
        excerpts.append(f"## {os.path.basename(path)}\n{' '.join(words[:SUMMARY_WORDS])}")
    log.info(
        "LLM request -> %s %s | summarise: %s",
        LLM_BASE_URL,
        LLM_MODEL,
        [os.path.basename(p) for p in files],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Give a short plain-language summary of each document below "
            "(a few bullet points per document), then one line on what they cover "
            "together.\n\n" + "\n\n".join(excerpts),
        },
    ]
    yield from stream_llm(messages)


def retrieve(question: str) -> list[tuple[str, dict]]:
    res = collection.query(query_embeddings=embed([question], "query"), n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(message: str, history: list[dict]):
    if collection.count() == 0:
        yield "No documents indexed yet — upload files and click Submit & Analyse first."
        return
    hits = retrieve(message)
    context = "\n\n".join(
        # 'p.' fallback: chunks indexed before Excel support stored {"page": N}
        f"[{m['source']} {m.get('loc', 'p.' + str(m.get('page', '?')))}]\n{doc}"
        for doc, m in hits
    )
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        # keep only the keys the OpenAI API accepts — gradio adds extras
        + [{"role": m["role"], "content": m["content"]} for m in history]
        + [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}]
    )
    log.info(
        "LLM request -> %s %s | Q: %s | context: %s",
        LLM_BASE_URL,
        LLM_MODEL,
        message,
        [f"{m['source']} {m.get('loc', m.get('page', '?'))}" for _, m in hits],
    )
    yield from stream_llm(messages)


CSS = """
.gradio-container {max-width: 760px !important; margin: 0 auto !important;}
#brand {text-align: center; margin-top: 16px;}
#brand p {color: var(--body-text-color-subdued);}
"""

with gr.Blocks(title="DocMagic") as demo:
    gr.Markdown(
        "# ✨ DocMagic\nChat with your Docs (supports PDFs and Excel)",
        elem_id="brand",
    )
    files = gr.File(
        file_count="multiple",
        file_types=[".pdf", ".xlsx", ".xlsm"],
        label=f"Drop up to {MAX_FILES} files here",
    )
    analyse_btn = gr.Button("Submit & Analyse", variant="primary")
    summary = gr.Markdown()
    analyse_btn.click(analyse, inputs=files, outputs=summary)
    with gr.Accordion("💬 Chat with your docs", open=False):
        gr.ChatInterface(answer)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
