"""DocMagic — chat with your Docs (PDFs and Excel). LangChain + Streamlit edition.

Same RAG pipeline as master, rebuilt on LangChain abstractions with a
Streamlit frontend: loaders -> splitter -> Chroma vector store -> LCEL chain.
Run: streamlit run app.py
"""

import logging
import os
import tempfile
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook

load_dotenv(override=True)  # .env edits win on rerun

# root stays at WARNING so httpx/streamlit chatter is hidden; only our logs show
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("docmagic")
log.setLevel(logging.INFO)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "missing-key-see-.env.example"
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or LLM_API_KEY
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

CHUNK_CHARS = 1500  # splitter counts characters; ~250 words
CHUNK_OVERLAP = 300  # ~50 words
TOP_K = 5
KB_TOP_K = 3  # extra chunks from the built-in logistics knowledge base (kb_ingest.py)
MAX_FILES = 2
SUMMARY_WORDS = 1500  # per file; keeps the summary prompt small and cheap

SYSTEM_PROMPT = (
    "You are DocMagic, an assistant for logistics, warehousing, ERP and CRM "
    "documents (SOPs, rate cards, contracts, manuals). Answer ONLY from the "
    "provided context. Cite the source like [file.pdf p.3] or [file.xlsx Rates] "
    "(sheet name for spreadsheets). "
    "The context may include general reference material labeled [Wikipedia ...]; "
    "prefer the user's documents for specifics and use the reference only for "
    "definitions and industry basics. "
    "If the context does not contain the answer, say so plainly."
)

# splits on paragraphs first, then lines, then words — chunks end at natural
# boundaries instead of mid-sentence (master's chunk_text cuts anywhere)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Give a short plain-language summary of each document below "
            "(a few bullet points per document), then one line on what they cover "
            "together.\n\n{excerpts}",
        ),
    ]
)


@st.cache_resource
def resources():
    """LLM + vector store, created once per server instead of on every rerun."""
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "nemotron" in LLM_MODEL else None
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.2,
        extra_body=extra,
    )
    # embed_documents() sends input_type=passage, embed_query() sends query —
    # the asymmetry we handled by hand in master's embed()
    embeddings = NVIDIAEmbeddings(
        base_url=LLM_BASE_URL, model=EMBED_MODEL, api_key=EMBED_API_KEY, truncate="END"
    )
    vectorstore = Chroma(
        collection_name="docs", embedding_function=embeddings, persist_directory="chroma_db"
    )
    # persistent logistics-basics collection; survives the per-Analyse wipe of "docs"
    kb_store = Chroma(  # chroma requires names >= 3 chars, so not "kb"
        collection_name="basics", embedding_function=embeddings, persist_directory="chroma_db"
    )
    return llm, vectorstore, kb_store


def load_units(path: str) -> list[Document]:
    """One Document per PDF page or Excel sheet, labeled for citations."""
    name = os.path.basename(path)
    if path.lower().endswith((".xlsx", ".xlsm")):
        wb = load_workbook(path, read_only=True, data_only=True)
        docs = []
        for ws in wb.worksheets:
            rows = (
                " | ".join(str(c) for c in row if c is not None)
                for row in ws.iter_rows(values_only=True)
            )
            text = "\n".join(r for r in rows if r)
            docs.append(Document(page_content=text, metadata={"source": name, "loc": ws.title}))
        wb.close()
        return docs
    return [
        Document(
            page_content=p.page_content,
            metadata={"source": name, "loc": f"p.{p.metadata.get('page', 0) + 1}"},
        )
        for p in PyPDFLoader(path).load()
    ]


def ingest(paths: list[str], vectorstore: Chroma) -> int:
    """Load, split and index the documents; returns the chunk count."""
    chunks = splitter.split_documents(doc for path in paths for doc in load_units(path))
    if chunks:
        vectorstore.add_documents(chunks)  # NVIDIAEmbeddings batches 50/call itself
    return len(chunks)


def format_docs(docs: list[Document]) -> str:
    log.info("context <- %s", [f"{d.metadata['source']} {d.metadata['loc']}" for d in docs])
    return "\n\n".join(
        f"[{d.metadata['source']} {d.metadata['loc']}]\n{d.page_content}" for d in docs
    )


def summary_stream(paths: list[str], llm):
    """Yield summary tokens for the first SUMMARY_WORDS words of each file."""
    excerpts = []
    for path in paths:
        words = []
        for doc in load_units(path):
            words += doc.page_content.split()
            if len(words) >= SUMMARY_WORDS:
                break
        excerpts.append(f"## {os.path.basename(path)}\n{' '.join(words[:SUMMARY_WORDS])}")
    log.info(
        "LLM request -> %s %s | summarise: %s",
        LLM_BASE_URL,
        LLM_MODEL,
        [os.path.basename(p) for p in paths],
    )
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    yield from chain.stream({"excerpts": "\n\n".join(excerpts)})


def answer_stream(question: str, history: list[dict], llm, vectorstore: Chroma, kb_store: Chroma):
    """Yield answer tokens from the LCEL chain: retrieve (docs + kb) -> prompt -> llm."""
    retrievers = {}
    if vectorstore.get(limit=1)["ids"]:
        retrievers["docs"] = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    if kb_store.get(limit=1)["ids"]:
        retrievers["kb"] = kb_store.as_retriever(search_kwargs={"k": KB_TOP_K})
    if not retrievers:
        yield "No documents indexed yet — upload files and click Submit & Analyse first."
        return
    merge = RunnableLambda(lambda hits: format_docs([d for v in hits.values() for d in v]))
    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | RunnableParallel(retrievers) | merge
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    log.info("LLM request -> %s %s | Q: %s", LLM_BASE_URL, LLM_MODEL, question)
    yield from chain.stream({"question": question, "history": history})


def main():
    st.set_page_config(page_title="DocMagic", page_icon="✨")
    llm, vectorstore, kb_store = resources()

    st.title("✨ DocMagic")
    st.caption("Chat with your Docs (supports PDFs and Excel)")
    files = st.file_uploader(
        f"Drop up to {MAX_FILES} files here",
        type=["pdf", "xlsx", "xlsm"],
        accept_multiple_files=True,
    )

    if st.button("Submit & Analyse", type="primary"):
        if not files:
            st.warning("Upload at least one file first.")
        elif len(files) > MAX_FILES:
            st.warning(f"Please upload at most {MAX_FILES} files at a time.")
        else:
            # uploads are in-memory; loaders want paths, so spill to a temp dir
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for f in files:
                    p = os.path.join(td, f.name)
                    with open(p, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(p)
                with st.spinner("Reading and indexing…"):
                    # each Submit & Analyse starts fresh so chat only covers current files
                    vectorstore.reset_collection()
                    n = ingest(paths, vectorstore)
                st.session_state.messages = []  # old chat referred to old docs
                st.session_state.summary = st.write_stream(summary_stream(paths, llm))
            st.caption(f"Indexed {n} chunks.")
    elif st.session_state.get("summary"):
        st.markdown(st.session_state.summary)

    st.divider()
    st.subheader("💬 Chat with your docs")
    for m in st.session_state.get("messages", []):
        st.chat_message(m["role"]).markdown(m["content"])
    if question := st.chat_input("Ask about your documents"):
        st.chat_message("user").markdown(question)
        history = st.session_state.get("messages", [])
        with st.chat_message("assistant"):
            text = st.write_stream(answer_stream(question, history, llm, vectorstore, kb_store))
        st.session_state.setdefault("messages", []).extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        )


if __name__ == "__main__":
    main()
