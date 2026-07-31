"""Promptable charts — a natural-language request becomes a Vega-Lite spec.

Why Vega-Lite rather than a hand-rolled renderer: its grammar already does
aggregation, binning, sorting and time units declaratively, so the LLM emits one
JSON object and no SQL has to be generated. The spec is data, not code.

The model never supplies the data — it only describes the encoding. We attach the
rows from Postgres ourselves, which is also what makes `_sanitize` sound: a spec
that cannot name a data source cannot fetch one.
"""

import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import load_frame, log, trace_config

MAX_ROWS = 5000  # ponytail: whole sheet to the browser; aggregate in SQL if a sheet outgrows this

CHART_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You turn a chart request into a Vega-Lite specification.\n"
            "Columns available (name and dtype): {columns}\n\n"
            "Reply with ONLY the JSON spec, no prose and no code fence. Rules:\n"
            '- Omit "data" and "$schema" entirely — they are attached for you.\n'
            '- Omit "width" and "height" — the container sizes the chart.\n'
            "- Use ONLY the exact column names listed above.\n"
            '- Aggregate in the encoding, e.g. {{"aggregate": "sum", "field": "Volume", '
            '"type": "quantitative"}} — never invent pre-aggregated columns.\n'
            "- Pick the mark that fits: temporal trend -> line or area, category "
            "comparison -> bar, share of a whole -> arc, two quantitative fields -> point.\n"
            '- Give every axis a readable "title", and add a top-level "title".\n'
            '- Sort bars by the measure, e.g. {{"sort": "-y"}}.\n'
            "- Never reference a URL, file or external dataset.",
        ),
        ("human", "{question}"),
    ]
)

CHART_HINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Recommend 2 or 3 useful chart questions for this uploaded spreadsheet. "
            "Reply with ONLY a JSON array of short questions. Use only the exact column names "
            "in the profile. Prefer meaningful trends, comparisons, distributions, or composition. "
            "Never recommend charts using serial numbers, row numbers, IDs, codes, invoice numbers, "
            "or any near-unique identifier. Do not invent columns. If no meaningful chart exists, return [].",
        ),
        ("human", "Sheet: {sheet}\nColumn profile: {profile}"),
    ]
)


def _sanitize(node):
    """Strip anything that could make the browser fetch a remote resource.

    The user's question reaches the model, so a crafted prompt could try to talk it
    into `data.url` or an image mark pointing at an attacker's host — which would
    leak the viewer's IP and any query string the model was told to include. The
    spec needs no URLs at all, so every `url` key goes, at every depth.
    """
    if isinstance(node, dict):
        return {k: _sanitize(v) for k, v in node.items() if k not in ("url", "data")}
    if isinstance(node, list):
        return [_sanitize(v) for v in node]
    return node


def _columns(df) -> str:
    return ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)


def _rows(df) -> list[dict]:
    """DataFrame → JSON-safe records (Timestamps to ISO, NaN to null)."""
    return json.loads(df.head(MAX_ROWS).to_json(orient="records", date_format="iso"))


def suggest_chart_hints(sid: str, sheet: str, llm) -> list[str]:
    """Generate safe, data-aware chart prompts for one uploaded spreadsheet sheet."""
    df = load_frame(sid, sheet)
    if df is None or df.empty:
        return []
    profile = [
        {
            "name": str(column),
            "dtype": str(df[column].dtype),
            "non_null": int(df[column].notna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
        }
        for column in df.columns
    ]
    try:
        hints = (CHART_HINT_PROMPT | llm | JsonOutputParser()).invoke(
            {"sheet": sheet, "profile": json.dumps(profile)}, config=trace_config()
        )
        if not isinstance(hints, list):
            return []
        return [hint.strip() for hint in hints if isinstance(hint, str) and hint.strip()][:3]
    except Exception:
        log.exception("chart hint generation failed | sheet=%s", sheet)
        return []


def build_chart(sid: str, sheet: str, question: str, llm) -> dict:
    """{spec, rows, truncated} for one prompted chart, or raises ValueError."""
    df = load_frame(sid, sheet)
    if df is None or df.empty:
        raise ValueError(f"No data for sheet '{sheet}'. Upload and analyse a spreadsheet first.")

    spec = (CHART_PROMPT | llm | JsonOutputParser()).invoke(
        {"columns": _columns(df), "question": question}, config=trace_config()
    )
    if not isinstance(spec, dict) or "mark" not in spec and "layer" not in spec:
        raise ValueError("I couldn't turn that into a chart — try naming a column and a measure.")

    spec = _sanitize(spec)
    # no $schema on purpose — the frontend passes mode:"vega-lite" to vega-embed, so the
    # bundled version is always the one used and a spec can never pin a mismatched one
    spec["data"] = {"values": _rows(df)}
    log.info("chart | sheet=%s mark=%s", sheet, spec.get("mark"))
    return {
        "spec": spec,
        "rows": min(len(df), MAX_ROWS),
        "truncated": len(df) > MAX_ROWS,
    }
