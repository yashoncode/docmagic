# DocMagic LangGraph implementation plan

Branch: `langgraph`. This is the current implementation; `langchain` remains the
feature-frozen Streamlit fallback.

## Objective

Provide grounded document chat and spreadsheet analysis while keeping each model
operation narrow, testable, and session-isolated.

## Implemented workflows

### Ingestion

```text
extract_metadata -> analyst -> reviewer -> end
                         ^         |
                         +-- revise once
```

- Metadata extracts PDF pages and Excel sheet columns/types.
- The analyst creates a source-grounded summary.
- The reviewer may request one correction; invalid reviewer output approves rather
  than blocking the upload.
- The completed result drives the summary and chart UI. Charts are available only
  after a successful upload result.

### Chat

```text
router -> documents | data | quant | web -> streamed answer
```

Each specialist receives only its relevant tools. Tool-incompatible models fall
back to the plain retrieval chain.

### Charts

`POST /api/chart` accepts a user prompt and selected uploaded sheet. The model emits
a Vega-Lite encoding using only supplied columns. The server strips `url` and `data`
keys recursively, then attaches rows from the current session's stored sheet. The
frontend derives chart prompts from that sheet's real typed columns, not hard-coded
business examples.

## Module ownership

| Module | Responsibility |
|---|---|
| `rag.py` | Models, Postgres, pgvector, extraction, retrieval, tools |
| `ingest_graph.py` | Upload metadata, analysis, review workflow |
| `chat_graph.py` | Question routing and specialist execution |
| `charts.py` | Vega-Lite prompting, validation, and data attachment |
| `api.py` | HTTP, SSE, session cookie, and error shaping |
| `web/` | Upload/chat workspace and post-upload analysis UI |

## Checks

```powershell
.\.venv\Scripts\python.exe test_rag.py
cd web
npm run lint
npm test
npm run build
```
