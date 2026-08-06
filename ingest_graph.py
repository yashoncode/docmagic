"""Ingestion pipeline — a LangGraph StateGraph that runs when documents are uploaded.

    extract_metadata → analyst → reviewer ─(approved)→ END
                          ↑         │
                          └(revise, once)┘

Each node is a small agent step. The reviewer grounds the analyst's summary against the
source excerpts before it reaches the user, and can bounce it back for exactly one revision
(bounded — no runaway critic loop).

build_ingest_graph(llm) closes over the LLM and returns the compiled graph; call
graph.astream({"paths": [...]}) to drive it and watch each node complete.
"""

import json
import os
from typing import TypedDict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from rag import SYSTEM_PROMPT, excel_frames, load_units, log, trace_config

EXCERPT_WORDS = 1200  # per file; keeps the analyst/reviewer prompts small and cheap
MAX_REVISIONS = 1  # reviewer can bounce the analysis back exactly once


class IngestState(TypedDict, total=False):
    paths: list[str]
    excerpts: str
    metadata: list[dict]
    analysis: str
    review: dict
    revisions: int


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify each logistics/ERP/CRM document by type. Reply with ONLY a JSON list, "
            "one object per document: "
            '[{{"source": "<filename>", "doc_type": "<SOP|rate card|contract|invoice|'
            'inventory report|manual|report|other>"}}]. Use the exact filenames from the '
            "'## <filename>' headers.",
        ),
        ("human", "{excerpts}"),
    ]
)

ANALYST_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "{revision_note}Document metadata (JSON): {metadata}\n\nExcerpts:\n{excerpts}\n\n"
            "Write, in markdown:\n"
            "1. A few plain-language bullets per document.\n"
            "2. A short **Key findings** list — the most useful concrete facts "
            "(rates, totals, dates, parties, quantities).\n"
            "3. One closing line on what the documents cover together.\n"
            "Ground EVERY claim in the excerpts — do not invent numbers, names or figures.",
        ),
    ]
)

REVIEWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a meticulous reviewer. Check the analysis is fully grounded in the source "
            "excerpts: no invented numbers, names, dates or claims. Reply with ONLY JSON: "
            '{{"approved": true|false, "notes": "<if not approved, name the specific ungrounded '
            'claims to fix; otherwise empty>"}}.',
        ),
        ("human", "Excerpts:\n{excerpts}\n\nAnalysis to review:\n{analysis}"),
    ]
)

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Suggest 3 questions the user could ask about THESE documents. Reply with ONLY a "
            "JSON array of short questions. Base every question on the actual content — name "
            "the real parties, sheets, routes, rates or dates that appear. No generic "
            "definition questions, no questions the documents can't answer.",
        ),
        ("human", "Metadata (JSON): {metadata}\n\nSummary of the documents:\n{analysis}"),
    ]
)


def suggest_questions(analysis: str, metadata: list[dict], llm) -> list[str]:
    """Document-aware chat starters. Best-effort — [] falls back to the generic ones."""
    try:
        questions = (QUESTION_PROMPT | llm | JsonOutputParser()).invoke(
            {"analysis": analysis, "metadata": json.dumps(metadata, default=str)},
            config=trace_config(),
        )
        if not isinstance(questions, list):
            return []
        return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:3]
    except Exception:
        log.exception("question suggestion failed")
        return []


def _excerpt(path: str) -> str:
    """First EXCERPT_WORDS words of a file, as a labelled block for the prompts."""
    words = []
    for doc in load_units(path):
        words += doc.page_content.split()
        if len(words) >= EXCERPT_WORDS:
            break
    return f"## {os.path.basename(path)}\n{' '.join(words[:EXCERPT_WORDS])}"


def _columns(df) -> list[str]:
    """Column names tagged with dtype, for the metadata the frontend shows."""
    return [f"{c} ({df[c].dtype})" for c in df.columns]


def build_ingest_graph(llm):
    """Compile the ingestion graph bound to `llm`."""

    def extract_metadata(state: IngestState) -> dict:
        meta, blocks = [], []
        for path in state["paths"]:
            name = os.path.basename(path)
            if path.lower().endswith((".xlsx", ".xlsm")):
                with open(path, "rb") as fh:
                    sheets = excel_frames(fh.read())
                meta.append(
                    {
                        "source": name,
                        "kind": "Excel",
                        "sheets": {
                            s: {"rows": len(df), "columns": _columns(df)}
                            for s, df in sheets.items()
                        },
                    }
                )
            else:
                meta.append({"source": name, "kind": "PDF", "pages": len(load_units(path))})
            blocks.append(_excerpt(path))
        excerpts = "\n\n".join(blocks)
        try:  # doc-type classification is best-effort — fall back to the file kind
            types = (CLASSIFY_PROMPT | llm | JsonOutputParser()).invoke(
                {"excerpts": excerpts}, config=trace_config()
            )
            by_source = {t.get("source"): t.get("doc_type") for t in types}
            for i, m in enumerate(meta):
                dt = by_source.get(m["source"])
                if not dt and len(types) == len(meta):  # LLM didn't echo the filename — trust order
                    dt = types[i].get("doc_type")
                m["doc_type"] = dt or m["kind"]
        except Exception:
            log.exception("doc-type classify failed — using file kind")
            for m in meta:
                m["doc_type"] = m["kind"]
        return {"excerpts": excerpts, "metadata": meta, "revisions": 0}

    def analyst(state: IngestState) -> dict:
        note = ""
        if state.get("revisions"):
            note = f"A reviewer flagged issues to fix: {state.get('review', {}).get('notes', '')}\n\n"
        log.info("ingest | analysing (revision %d)", state.get("revisions", 0))
        analysis = (ANALYST_PROMPT | llm | StrOutputParser()).invoke(
            {
                "revision_note": note,
                "metadata": json.dumps(state["metadata"], default=str),
                "excerpts": state["excerpts"],
            },
            config=trace_config(),
        )
        return {"analysis": analysis}

    def reviewer(state: IngestState) -> dict:
        try:
            review = (REVIEWER_PROMPT | llm | JsonOutputParser()).invoke(
                {"excerpts": state["excerpts"], "analysis": state["analysis"]},
                config=trace_config(),
            )
        except Exception:  # unparseable review → don't block the user
            log.exception("review failed to parse — approving")
            review = {"approved": True, "notes": ""}
        revisions = state.get("revisions", 0) + (0 if review.get("approved") else 1)
        log.info("ingest | review approved=%s revisions=%d", review.get("approved"), revisions)
        return {"review": review, "revisions": revisions}

    def after_review(state: IngestState) -> str:
        if state["review"].get("approved") or state.get("revisions", 0) > MAX_REVISIONS:
            return "done"
        return "analyst"

    g = StateGraph(IngestState)
    g.add_node("extract_metadata", extract_metadata)
    g.add_node("analyst", analyst)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "extract_metadata")
    g.add_edge("extract_metadata", "analyst")
    g.add_edge("analyst", "reviewer")
    g.add_conditional_edges("reviewer", after_review, {"analyst": "analyst", "done": END})
    return g.compile()
