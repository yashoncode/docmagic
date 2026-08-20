"""Chat supervisor — a router sends each turn to ONE capability specialist.

Specialists differ by the TOOLS they hold, not by persona:
    documents → search_documents         (RAG over the uploaded files)
    data      → describe_table, calculate (computed stats over Excel sheets)
    quant     → calculate, search_documents, reconcile_invoice (numbers, then the math)
    web       → search_web                (Tavily, only offered when a key is set)

route() classifies the turn (with an arithmetic fast-path), then the chosen specialist —
a LangGraph create_agent holding only its lane's tools — streams the answer. Models without
tool support fall back to the plain LCEL RAG chain, so chat never hard-fails.

Streaming contract (consumed by api.chat): an async generator yielding answer text as
plain strings, interleaved with dicts — {"route": lane} once, then {"tool": name} per
tool call. Labels and icons are the frontend's job; this layer emits names only.
"""

import asyncio
import json
import re
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from openai import AuthenticationError

from rag import (
    AGENT_PROMPT,
    SYSTEM_PROMPT,
    format_docs,
    load_frame,
    log,
    reconcile,
    retrieve_docs,
    safe_calc,
    sheet_names,
    trace_config,
    web_documents,
)

LANE_PROMPTS = {
    "documents": SYSTEM_PROMPT
    + " Answer from the user's uploaded documents: call search_documents, then cite "
    "[file p.N] or [file sheet].",
    "web": SYSTEM_PROMPT
    + " Answer from live web results: call search_web and cite sources by domain.",
    "quant": AGENT_PROMPT
    + " This turn needs calculation. Fetch the numbers with search_documents first, then use "
    "calculate for EVERY arithmetic step — never do math in your head. Show the working. "
    "If the user is checking a bill, invoice or charge against agreed/contracted rates, "
    "pull each line's rate, quantity and charged amount from the documents and call "
    "reconcile_invoice — report its verdict as given, never overrule it.",
    "data": SYSTEM_PROMPT
    + " The user is asking about spreadsheet data. Use describe_table for real computed stats "
    "on the named sheet, and calculate for any further arithmetic. Cite the sheet name.",
}

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Route the user's message to ONE specialist. Reply with ONLY the label, nothing else.\n"
            "documents — about the user's uploaded files (PDFs, contracts, SOPs, rate cards)\n"
            "data — totals, averages, max/min, counts or trends over uploaded spreadsheet columns\n"
            "quant — arithmetic, rates, GST, percentages, multi-number math, or checking an "
            "invoice/bill against a rate card or contracted rate\n"
            "web — general knowledge or current/external information\n"
            "Available labels: {labels}. If unsure, pick documents.",
        ),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

_ARITHMETIC = re.compile(r"^[\d\s+\-*/().%]+$")


def _build_tools(lane: str, ctx: dict) -> list:
    """The tools for one lane, closed over this turn's context."""
    from langchain_core.tools import tool

    vectorstore, embed_key = ctx["vectorstore"], ctx["embed_key"]
    tavily_key, sid = ctx["tavily_key"], ctx["sid"]
    step = ctx.get("progress") or (lambda _stage: None)

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents (PDFs and spreadsheets)."""
        hits = retrieve_docs(query, vectorstore, embed_key, progress=step)
        step("composing")
        return format_docs(hits) if hits else "No documents are indexed yet."

    @tool
    def calculate(expression: str) -> str:
        """Do arithmetic exactly — rates, totals, percentages, GST. E.g. '500 * 12 * 0.18'."""
        try:
            return str(safe_calc(expression))
        except Exception:
            return f"Couldn't evaluate '{expression}' — use plain arithmetic like 500*12*0.18."

    @tool
    def reconcile_invoice(lines: str) -> str:
        """Check billed amounts against contracted rates. Fetch the rate-card rate AND the
        invoiced amount with search_documents first, then pass a JSON list:
        [{"item": "Chennai haulage", "rate": 1200, "qty": 3, "charged": 3800}]."""
        try:
            result = reconcile(json.loads(lines))
        except json.JSONDecodeError:
            return 'Pass a JSON list like [{"item": "...", "rate": 0, "qty": 0, "charged": 0}].'
        except ValueError as e:
            return f"Couldn't reconcile: {e}"
        if not result["lines"]:
            return "No lines to reconcile — extract the rate, quantity and charged amount first."
        rows = "\n".join(
            f"{r['item']}: {r['rate']} x {r['qty']} = {r['expected']} expected, "
            f"{r['charged']} charged, delta {r['delta']:+} ({r['status']})"
            for r in result["lines"]
        )
        step("composing")
        return f"{rows}\n\n{result['mismatches']} mismatch(es), net variance {result['variance']:+}"

    @tool
    def search_web(query: str) -> str:
        """Search the web for general knowledge, logistics concepts or current info."""
        docs = web_documents(query, tavily_key)
        step("composing")
        return format_docs(docs) if docs else "No web results found."

    @tool
    def describe_table(sheet: str) -> str:
        """Computed stats (totals, mean, min, max, counts) for one uploaded spreadsheet sheet."""
        step("reading_sheet")
        df = load_frame(sid, sheet)
        if df is None:
            names = sheet_names(sid)
            return f"No sheet named '{sheet}'. Available sheets: {', '.join(names) or 'none'}."
        num = df.select_dtypes("number")
        if num.empty:
            return f"Sheet '{sheet}' has no numeric columns; {len(df)} rows, columns: {', '.join(map(str, df.columns))}."
        stats = num.describe().round(2).to_string()
        totals = num.sum().round(2).to_string()
        step("composing")
        return f"Sheet '{sheet}' — {len(df)} rows.\nColumn totals:\n{totals}\n\nStatistics:\n{stats}"

    lane_tools = {
        "documents": [search_documents],
        "web": [search_web] if tavily_key else [search_documents],
        "quant": [calculate, search_documents, reconcile_invoice],
        "data": [describe_table, calculate],
    }
    return lane_tools[lane]


async def route(question: str, llm, labels: list[str]) -> str:
    """Pick a specialist lane for this turn (arithmetic fast-path, then LLM classifier)."""
    if _ARITHMETIC.match(question.strip()):  # pure sum like "500*12*0.18" — no LLM needed
        return "quant" if "quant" in labels else "documents"
    try:
        raw = await (ROUTER_PROMPT | llm | StrOutputParser()).ainvoke(
            {"question": question, "labels": ", ".join(labels)}, config=trace_config()
        )
    except Exception:
        log.exception("router failed — defaulting to documents")
        return "documents"
    raw = raw.strip().lower()
    for lane in labels:  # tolerate extra words around the label
        if lane in raw:
            return lane
    return "documents"


async def _specialist_tokens(lane: str, question: str, history: list[dict], llm, ctx: dict):
    """Stream one specialist agent's tokens, with a {"tool": name} dict per tool call."""
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessageChunk

    agent = create_agent(llm, _build_tools(lane, ctx), system_prompt=LANE_PROMPTS[lane])
    async for chunk, _ in agent.astream(
        {"messages": [*history, {"role": "user", "content": question}]},
        stream_mode="messages",
        config=trace_config(),
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        for tc in chunk.tool_call_chunks or []:
            if tc.get("name"):
                yield {"tool": tc["name"]}
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content


async def answer_stream(question: str, history: list[dict], llm, ctx: dict):
    """Plain LCEL fallback for models without tool support: retrieve (docs + web) → prompt → llm."""
    vectorstore, embed_key, tavily_key = ctx["vectorstore"], ctx["embed_key"], ctx["tavily_key"]
    retrievers = {}
    if ctx.get("has_docs"):
        retrievers["docs"] = RunnableLambda(lambda q: retrieve_docs(q, vectorstore, embed_key))
    if tavily_key:
        retrievers["web"] = RunnableLambda(lambda q: web_documents(q, tavily_key))
    if not retrievers:
        yield "No documents indexed yet — upload a file, or turn on web search."
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
    log.info("LLM request (plain chain) | Q: %s", question)
    async for token in chain.astream(
        {"question": question, "history": history}, config=trace_config()
    ):
        yield token


_END = object()


async def _with_progress(stream, stages: asyncio.Queue):
    """Yield the specialist's events, interleaved with stage updates as they land.

    Tools run in worker threads, so their `progress()` calls arrive on a queue rather
    than in the token stream. Pumping both through one queue keeps the UI live during
    a slow retrieval instead of silent until the next token.
    """

    async def pump():
        try:
            async for item in stream:
                await stages.put(item)
        except Exception as e:  # re-raised below: the caller's fallback depends on seeing it
            await stages.put(e)
        finally:
            await stages.put(_END)

    task = asyncio.create_task(pump())
    try:
        while True:
            item = await stages.get()
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def chat_stream(question: str, history: list[dict], llm, ctx: dict):
    """Route to a specialist and stream it; fall back to the plain chain for non-tool models.

    ctx: vectorstore, embed_key, tavily_key, sid, has_docs, has_data.
    """
    labels = ["documents", "quant"]
    if ctx.get("has_data"):
        labels.append("data")
    if ctx.get("tavily_key"):
        labels.append("web")

    yield {"stage": "routing"}
    lane = await route(question, llm, labels)
    log.info("router → %s | Q: %s", lane, question)
    yield {"route": lane}

    # tools call progress() from worker threads; hop back to this loop to enqueue
    stages: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    ctx = {**ctx, "progress": lambda stage: loop.call_soon_threadsafe(
        stages.put_nowait, {"stage": stage}
    )}

    stream = _with_progress(_specialist_tokens(lane, question, history, llm, ctx), stages)
    try:
        first = await anext(stream)  # non-tool models error here, before any output
    except StopAsyncIteration:
        return
    except AuthenticationError:
        raise  # bad API key — the plain chain would fail the same way, so surface it
    except Exception:
        log.exception("agent unavailable — falling back to the plain RAG chain")
        async for token in answer_stream(question, history, llm, ctx):
            yield token
        return
    yield first
    async for event in stream:
        yield event
