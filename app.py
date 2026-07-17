"""DocMagic / XEON AI — chat with your Docs (PDFs and Excel). LangChain + Streamlit edition.

RAG pipeline on LangChain abstractions with a Streamlit frontend:
loaders -> splitter -> Chroma vector store -> LCEL chain.
Run: streamlit run app.py
"""

import io
import logging
import os
import tempfile
import time
from operator import itemgetter

import pandas as pd
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
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or LLM_API_KEY
# embeddings keep their own endpoint so switching LLM provider can't break them
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

CHUNK_CHARS = 1500  # splitter counts characters; ~250 words
CHUNK_OVERLAP = 300  # ~50 words
TOP_K = 5
KB_TOP_K = 3  # extra chunks from the built-in logistics knowledge base
MAX_FILES = 2
SUMMARY_WORDS = 1500  # per file; keeps the summary prompt small and cheap
CHART_SERIES = 3  # max numeric series per auto-chart; more gets unreadable

CURATED_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "qwen/qwen3-next-80b-a3b-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "deepseek-ai/deepseek-v4-flash",
]
CUSTOM_MODEL = "Custom…"

SUGGESTIONS = {
    ":blue[:material/help:] What is cross-docking?": "What is cross-docking?",
    ":orange[:material/school:] Incoterms in simple terms": "Explain Incoterms in simple terms",
    ":violet[:material/description:] What's in my documents?": (
        "Give me an overview of what my documents contain"
    ),
}

SYSTEM_PROMPT = (
    "You are XEON AI, a friendly expert on logistics, warehousing, ERP and CRM "
    "documents (SOPs, rate cards, contracts, manuals). Talk like a helpful "
    "colleague: warm, plain language, short sentences — never stiff or robotic. "
    "For questions about the documents, answer ONLY from the provided context and "
    "cite the source like [file.pdf p.3] or [file.xlsx Rates] (sheet name for "
    "spreadsheets). The context may include general reference material labeled "
    "[Wikipedia ...]; prefer the user's documents for specifics and use the "
    "reference only for definitions and industry basics. If the context does not "
    "have the answer, say so in one friendly sentence — do not lecture about what "
    "the context contains. "
    "If the user is just greeting or reacting ('hi', 'nice', 'thanks'), reply "
    "warmly in a sentence or two — no citations, no mention of context or documents."
)

# splits on paragraphs first, then lines, then words — chunks end at natural
# boundaries instead of mid-sentence
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


def secret(name: str) -> str:
    """st.secrets first (Streamlit Cloud), then environment (.env locally)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml anywhere
        pass
    return os.getenv(name, "")


@st.cache_resource(max_entries=4)
def resources(llm_key: str, llm_base_url: str, llm_model: str, embed_key: str):
    """LLM + vector stores, cached per (key, model) so sidebar changes take effect."""
    kwargs = {}
    if "nemotron" in llm_model:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if "kimi" not in llm_model:
        kwargs["temperature"] = 0.2  # kimi-k3 fixes temperature=1.0 and rejects overrides
    llm = ChatOpenAI(base_url=llm_base_url, api_key=llm_key, model=llm_model, **kwargs)
    # embed_documents() sends input_type=passage, embed_query() sends query
    embeddings = NVIDIAEmbeddings(
        base_url=EMBED_BASE_URL, model=EMBED_MODEL, api_key=embed_key, truncate="END"
    )
    vectorstore = Chroma(
        collection_name="docs", embedding_function=embeddings, persist_directory="chroma_db"
    )
    # persistent logistics-basics collection; survives the per-Analyse wipe of "docs"
    kb_store = Chroma(  # chroma requires names >= 3 chars, so not "kb"
        collection_name="basics", embedding_function=embeddings, persist_directory="chroma_db"
    )
    return llm, vectorstore, kb_store


def default_resources():
    """Env-configured resources, for scripts like kb_ingest.py."""
    return resources(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMBED_API_KEY)


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
    log.info("LLM request | summarise: %s", [os.path.basename(p) for p in paths])
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
        yield "No documents indexed yet — upload files and click Submit & analyse first."
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
    log.info("LLM request | Q: %s", question)
    yield from chain.stream({"question": question, "history": history})


@st.cache_data(max_entries=8)
def excel_frames(data: bytes) -> dict[str, pd.DataFrame]:
    """All sheets of an uploaded workbook as DataFrames, keyed by sheet name."""
    return pd.read_excel(io.BytesIO(data), sheet_name=None)


def sheet_chart(df: pd.DataFrame):
    """Auto-chart one sheet: time series -> line, categories -> bar, else index line."""
    df = df.dropna(axis=1, how="all")
    num_cols = list(df.select_dtypes("number").columns)[:CHART_SERIES]
    if not num_cols:
        st.caption("No numeric columns to chart in this sheet.")
        return
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        st.line_chart(df, x=date_cols[0], y=num_cols)
        return
    cat_cols = [
        c for c in df.columns if c not in num_cols and df[c].dtype == object and df[c].nunique() <= 30
    ]
    if cat_cols:
        top = (
            df.groupby(cat_cols[0])[num_cols]
            .sum()
            .sort_values(num_cols[0], ascending=False)
            .head(12)
        )
        st.bar_chart(top, y=num_cols)
        return
    st.line_chart(df[num_cols])


def render_analysis(files):
    """Charts + previews for uploaded Excel files (analysis toggle)."""
    st.header(":material/query_stats: Data analysis")
    excel = [f for f in files if f.name.lower().endswith((".xlsx", ".xlsm"))]
    if not excel:
        st.caption("Charts need Excel files — PDFs are text-only.")
        return
    for f in excel:
        st.subheader(f.name)
        sheets = excel_frames(f.getvalue())
        tabs = st.tabs(list(sheets))
        for tab, (name, df) in zip(tabs, sheets.items()):
            with tab:
                st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
                sheet_chart(df)
                with st.expander("Data preview", icon=":material/table_rows:"):
                    st.dataframe(df.head(100))


def main():
    st.set_page_config(page_title="XEON AI — DocMagic", page_icon="✨")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("summary", "")

    with st.sidebar:
        st.subheader(":material/key: Access")
        pasted = st.text_input(
            "API key",
            type="password",
            placeholder="nvapi-…",
            help="One free NVIDIA key (build.nvidia.com) powers chat and embeddings. "
            "Leave blank if the app is deployed with its own key.",
        )
        model = st.selectbox(
            "Chat model",
            list(dict.fromkeys([LLM_MODEL, *CURATED_MODELS])) + [CUSTOM_MODEL],
            help="All options run on NVIDIA NIM's free tier.",
        )
        if model == CUSTOM_MODEL:
            model = st.text_input("Model id", placeholder="provider/model-name").strip() or LLM_MODEL
        with st.expander("Advanced", icon=":material/tune:"):
            base_url = st.text_input("LLM base URL", value=LLM_BASE_URL)

        st.subheader(":material/folder: Documents")
        files = st.file_uploader(
            f"Up to {MAX_FILES} files",
            type=["pdf", "xlsx", "xlsm"],
            accept_multiple_files=True,
        )
        analyse = st.button(
            "Submit & analyse", type="primary", icon=":material/auto_awesome:", width="stretch"
        )
        analysis_on = st.toggle(
            "Enable analysis", help="Charts and data previews for uploaded Excel files"
        )

    st.title("✨ XEON AI")
    st.caption("Chat with your documents — answers cited to the exact file, page or sheet.")

    llm_key = pasted or secret("LLM_API_KEY")
    if not llm_key:
        st.info("Paste your free NVIDIA API key in the sidebar to get started.", icon=":material/key:")
        st.stop()
    embed_key = pasted or secret("EMBED_API_KEY") or llm_key
    llm, vectorstore, kb_store = resources(llm_key, base_url, model, embed_key)

    # back to the sidebar for index status + one-click basics seeding (needs resources)
    with st.sidebar:
        kb_n = kb_store._collection.count()
        st.caption(f"Index: {vectorstore._collection.count():,} document chunks · {kb_n:,} basics chunks")
        if kb_n == 0 and st.button(
            "Seed logistics basics",
            icon=":material/school:",
            width="stretch",
            help="Download ~16 Wikipedia articles so XEON AI knows industry basics",
        ):
            from kb_ingest import ARTICLES, fetch

            with st.status("Downloading logistics basics…") as status:
                units = []
                for title in ARTICLES:
                    time.sleep(1)  # stay under wikipedia's rate limit
                    text = fetch(title)
                    if text:
                        units.append(
                            Document(page_content=text, metadata={"source": "Wikipedia", "loc": title})
                        )
                        status.write(title)
                kb_store.add_documents(splitter.split_documents(units))
                status.update(label="Basics indexed", state="complete")
            st.rerun()

    if analyse:
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
                    # each Submit & analyse starts fresh so chat only covers current files
                    vectorstore.reset_collection()
                    n = ingest(paths, vectorstore)
                st.session_state.messages = []  # old chat referred to old docs
                st.toast(f"Indexed {n} chunks", icon=":material/check_circle:")
                with st.expander("Document summary", icon=":material/description:", expanded=True):
                    st.session_state.summary = st.write_stream(summary_stream(paths, llm))
    elif st.session_state.summary:
        with st.expander("Document summary", icon=":material/description:"):
            st.markdown(st.session_state.summary)

    if analysis_on and files:
        render_analysis(files)

    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    question = None
    if not st.session_state.messages:
        picked = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
        if picked:
            question = SUGGESTIONS[picked]
    prompt = st.chat_input("Ask about your documents", submit_mode="disable")
    question = prompt or question

    if question:
        st.chat_message("user").markdown(question)
        try:
            with st.chat_message("assistant"):
                text = st.write_stream(
                    answer_stream(question, st.session_state.messages, llm, vectorstore, kb_store)
                )
        except Exception as e:  # rate limits & co: tell the user, keep history clean
            st.error(f"Model call failed — try again in a moment. ({str(e)[:120]})", icon=":material/error:")
        else:
            st.session_state.messages.extend(
                [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
            )


if __name__ == "__main__":
    main()
