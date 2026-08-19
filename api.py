"""DocMagic HTTP API — FastAPI. The Next.js frontend is the only client.

Thin by design: session cookie, SSE encoding, and error shaping. All the thinking
lives in rag.py (retrieval, storage) and the two graphs (chat_graph, ingest_graph).

Two endpoints stream Server-Sent Events:
  POST /api/upload   ingested → node* → result
  POST /api/chat     status* / token* → done | error

Signed in with Google (see auth.py); every model call is billed against the
user's token credits, so all four model endpoints require a session.

Env: POSTGRES_URL (required), EMBED_API_KEY, LLM_API_KEY, TAVILY_API_KEY,
     GOOGLE_CLIENT_ID, SESSION_SECRET, FREE_TOKENS, ADMIN_EMAILS,
     WEB_ORIGIN (frontend origin for CORS), COOKIE_CROSS_SITE=1 when the frontend
     is on a different site than this API.

Run: uvicorn api:app --reload --port 8000
"""

import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks import UsageMetadataCallbackHandler
from openai import AuthenticationError
from pydantic import BaseModel
from sqlalchemy import text
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

import auth
from charts import build_chart, suggest_chart_hints
from chat_graph import chat_stream
from ingest_graph import build_ingest_graph, suggest_questions
from rag import (
    EMBED_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    docs_store,
    engine,
    excel_frames,
    ingest,
    load_frame,
    log,
    purge,
    reset_store,
    reset_uploads,
    resources,
    save_frames,
    sheet_names,
    trace_config,
)

MAX_FILES = 2
SESSION_DAYS = 30

SUGGESTIONS = [
    "What is cross-docking?",
    "Explain Incoterms in simple terms",
    "Give me an overview of what my documents contain",
]

# cross-site cookies need SameSite=None + Secure; localhost dev wants plain Lax
CROSS_SITE = os.getenv("COOKIE_CROSS_SITE", "") == "1"
# rstrip: an origin is scheme+host+port and never ends in "/", but pasting a browser URL
# into the host's env panel does — and Starlette compares the string exactly, so the
# stray slash rejects every real request with "Disallowed CORS origin"
WEB_ORIGIN = os.getenv("WEB_ORIGIN", "http://localhost:3000").rstrip("/")

# warn rather than raise: the suite boots the app without a database, and the first
# real request fails loudly anyway
if not os.getenv("POSTGRES_URL"):
    log.warning("POSTGRES_URL is not set - uploads and chat will fail. See .env.example")

app = FastAPI(title="DocMagic API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[WEB_ORIGIN],  # explicit origin: credentialed CORS forbids "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cookie(response, name: str, value: str, days: int = SESSION_DAYS) -> None:
    """Our cookies, one set of flags. Cross-site needs SameSite=None + Secure."""
    response.set_cookie(
        name,
        value,
        httponly=True,
        samesite="none" if CROSS_SITE else "lax",
        secure=CROSS_SITE,
        max_age=days * 24 * 3600,
    )


@app.middleware("http")
async def session_cookie(request: Request, call_next):
    """Mint/carry the session id. Middleware, not a dependency, so SSE responses get it too.

    Documents stay keyed to the browser session, never to the account: the user row
    holds the token ledger and nothing else, and signing out erases the uploads.
    Cookie parsing only — the user row is looked up per route, where a database
    outage can still be reported properly.
    """
    sid = request.cookies.get("sid") or uuid4().hex[:12]
    request.state.sid = sid
    request.state.uid = auth.parse(request.cookies.get(auth.COOKIE, ""))
    response = await call_next(request)
    _cookie(response, "sid", sid)
    return response


async def signed_in(request: Request) -> dict:
    """The current user, or 401. Every model call goes through here."""
    if not request.state.uid:
        raise HTTPException(401, "Please sign in to continue.")
    try:
        user = await run_in_threadpool(auth.user, request.state.uid)
    except RuntimeError as e:  # database missing/unreachable
        raise HTTPException(503, str(e)) from e
    if not user:
        raise HTTPException(401, "Please sign in to continue.")
    return user


async def with_credits(request: Request) -> dict:
    user = await signed_in(request)
    if user["tokens_left"] <= 0:
        raise HTTPException(
            402, "You're out of token credits. Ask an admin to top up your account."
        )
    return user


def metered(llm):
    """(llm that reports its token usage, the handler holding the totals).

    `resources()` is cached across requests, so the callback goes on a copy —
    never on the shared model.
    """
    handler = UsageMetadataCallbackHandler()
    return llm.model_copy(update={"callbacks": [handler]}), handler


async def bill(user: dict, handler, chars: int = 0) -> None:
    try:
        await run_in_threadpool(auth.spend, user["id"], auth.charged(handler, chars))
    except Exception:
        log.exception("could not bill tokens for %s", user["email"])  # never fail the answer


def sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


def _keys(pasted: str) -> tuple[str, str]:
    """(chat key, embed key). Embeddings run on the app's key so search stays free."""
    chat_key = pasted or LLM_API_KEY
    return chat_key, (EMBED_API_KEY or pasted or LLM_API_KEY)


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/config")
async def config():
    """Everything the frontend needs to render its controls."""
    return {
        "defaultModel": LLM_MODEL,
        "baseUrl": LLM_BASE_URL,
        "needsKey": not LLM_API_KEY,
        "dbReady": bool(os.getenv("POSTGRES_URL")),
        "maxFiles": MAX_FILES,
        "suggestions": SUGGESTIONS,
        "googleClientId": auth.GOOGLE_CLIENT_ID,
        "freeTokens": auth.FREE_TOKENS,
        "minPassword": auth.MIN_PASSWORD,
    }


class GoogleBody(BaseModel):
    credential: str  # the ID token from Google Identity Services


@app.post("/api/auth/google")
async def auth_google(request: Request, body: GoogleBody, response: Response):
    """Verify Google's ID token, create the account on first sign-in, start a session."""
    try:
        claims = await run_in_threadpool(auth.verify_google, body.credential)
        user = await run_in_threadpool(auth.login, claims)
    except ValueError as e:
        raise HTTPException(401, str(e)) from e
    except RuntimeError as e:  # sign-in or database not configured
        raise HTTPException(503, str(e)) from e
    _cookie(response, auth.COOKIE, auth.sign(user["id"]), auth.SESSION_DAYS)
    return user


class EmailBody(BaseModel):
    email: str
    password: str
    name: str = ""  # sign-up only


@app.post("/api/auth/signup")
async def auth_signup(body: EmailBody, response: Response):
    """Create an email account and start a session. No verification mail — the address
    is only an identifier here, so an unverified one costs nothing but its own credits."""
    return await _email_session(auth.signup, body, response)


@app.post("/api/auth/login")
async def auth_login(body: EmailBody, response: Response):
    return await _email_session(auth.login_email, body, response)


async def _email_session(action, body: EmailBody, response: Response) -> dict:
    args = (body.email, body.password) + ((body.name,) if action is auth.signup else ())
    try:
        user = await run_in_threadpool(action, *args)
    except ValueError as e:  # taken email, weak password, or wrong credentials
        raise HTTPException(400 if action is auth.signup else 401, str(e)) from e
    except RuntimeError as e:  # database not configured
        raise HTTPException(503, str(e)) from e
    _cookie(response, auth.COOKIE, auth.sign(user["id"]), auth.SESSION_DAYS)
    return user


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    """End the session and erase its uploads — documents are never kept after sign-out."""
    try:
        await run_in_threadpool(purge, request.state.sid)
    except Exception:
        log.exception("could not purge %s on logout", request.state.sid)  # still sign them out
    response.delete_cookie(auth.COOKIE, samesite="none" if CROSS_SITE else "lax", secure=CROSS_SITE)
    return {"ok": True}


@app.get("/api/me")
async def me(request: Request):
    """The signed-in user, or null. 200 either way — "not signed in" isn't an error."""
    if not request.state.uid:
        return {"user": None}
    try:
        return {"user": await run_in_threadpool(auth.user, request.state.uid)}
    except RuntimeError:
        return {"user": None}


async def admin(request: Request) -> dict:
    user = await signed_in(request)
    if not user["is_admin"]:
        raise HTTPException(403, "Admins only.")
    return user


@app.get("/api/admin/users")
async def admin_users(request: Request):
    await admin(request)
    return {"users": await run_in_threadpool(auth.list_users)}


class TokensBody(BaseModel):
    tokens: int


@app.post("/api/admin/users/{uid}/tokens")
async def admin_set_tokens(request: Request, uid: int, body: TokensBody):
    await admin(request)
    user = await run_in_threadpool(auth.set_tokens, uid, body.tokens)
    if not user:
        raise HTTPException(404, "No such user.")
    return user


@app.post("/api/upload")
async def upload(
    request: Request,
    files: list[UploadFile] = File(...),
    key: str = Form(""),
    model: str = Form(LLM_MODEL),
    base_url: str = Form(LLM_BASE_URL),
):
    """Index the uploads, extract their sheets, then stream the ingestion graph."""
    user = await with_credits(request)
    sid = request.state.sid
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Please upload at most {MAX_FILES} files at a time.")
    chat_key, embed_key = _keys(key)
    if not embed_key:
        raise HTTPException(400, "No embedding key configured — add EMBED_API_KEY or paste a key.")
    # read now: UploadFile is closed once the request handler returns, before the stream runs
    blobs = [(os.path.basename(f.filename or "file"), await f.read()) for f in files]

    async def events():
        usage = None
        try:
            llm, embeddings = resources(chat_key or "no-chat-key", base_url, model, embed_key)
            llm, usage = metered(llm)
            store = docs_store(embeddings, sid)
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for name, data in blobs:
                    path = os.path.join(td, name)
                    Path(path).write_bytes(data)
                    paths.append(path)

                # each upload starts fresh, so chat only ever covers the current files
                yield sse("node", {"name": "clearing"})
                await run_in_threadpool(reset_store, store)
                await run_in_threadpool(reset_uploads, sid)
                yield sse("node", {"name": "indexing"})
                chunks = await run_in_threadpool(ingest, paths, store)

                # tabular sheets land in Postgres for the data specialist and Vega-Lite charts
                for name, data in blobs:
                    if name.lower().endswith((".xlsx", ".xlsm")):
                        yield sse("node", {"name": "tables"})
                        await run_in_threadpool(save_frames, sid, name, excel_frames(data))

                yield sse("ingested", {"chunks": chunks})
                if chunks == 0:
                    yield sse("result", {"reason": "no_text"})
                    return
                if not chat_key:
                    yield sse("result", {"reason": "no_key"})
                    return

                final = {}
                async for update in build_ingest_graph(llm).astream(
                    {"paths": paths}, config=trace_config()
                ):
                    for node, out in update.items():
                        yield sse("node", {"name": node})
                        final.update(out or {})
                sheets = await run_in_threadpool(sheet_names, sid)
                if sheets:
                    yield sse("node", {"name": "chart_hints"})
                chart_hints = {
                    sheet: await run_in_threadpool(suggest_chart_hints, sid, sheet, llm)
                    for sheet in sheets
                }
                yield sse("node", {"name": "questions"})
                questions = await run_in_threadpool(
                    suggest_questions,
                    final.get("analysis", ""),
                    final.get("metadata", []),
                    llm,
                )
                yield sse(
                    "result",
                    {
                        "summary": final.get("analysis", ""),
                        "metadata": final.get("metadata", []),
                        "review": final.get("review", {}),
                        "sheets": sheets,
                        "chartHints": chart_hints,
                        "suggestions": questions,
                    },
                )
        except AuthenticationError:
            log.exception("upload: chat key rejected")
            yield sse("error", {"code": "auth", "message": _AUTH_MESSAGE})
        except RuntimeError as e:
            # a config problem, not a bad document — say so, or the user hunts the wrong bug
            log.exception("upload blocked by configuration")
            yield sse("error", {"code": "config", "message": str(e)})
        except Exception:
            log.exception("upload failed")
            yield sse(
                "error",
                {"message": "Couldn't read those documents — check the files and try again."},
            )
        finally:
            if usage:
                await bill(user, usage)

    return EventSourceResponse(events())


_AUTH_MESSAGE = (
    "Your chat API key was rejected. Check it's a valid key for the selected model "
    "(NVIDIA keys start with `nvapi-`) and try again."
)


class ChatBody(BaseModel):
    question: str
    history: list[dict] = []
    model: str = LLM_MODEL
    baseUrl: str = LLM_BASE_URL
    key: str = ""
    webOn: bool = False
    # the client knows whether its upload indexed; the alternative is querying pgvector's
    # internals. Worst case a lie costs one empty retrieval.
    hasDocs: bool = False


@app.post("/api/chat")
async def chat(request: Request, body: ChatBody):
    user = await with_credits(request)
    sid = request.state.sid
    chat_key, embed_key = _keys(body.key)
    if not chat_key:
        raise HTTPException(400, "Add an API key to chat about your documents.")
    llm, embeddings = resources(chat_key, body.baseUrl, body.model, embed_key)
    llm, usage = metered(llm)
    try:
        ctx = {
            "vectorstore": docs_store(embeddings, sid),
            "embed_key": embed_key,
            "tavily_key": os.getenv("TAVILY_API_KEY", "") if body.webOn else "",
            "sid": sid,
            "has_docs": body.hasDocs,
            # authoritative — it's the same lookup the describe_table tool will do
            "has_data": bool(await run_in_threadpool(sheet_names, sid)),
        }
    except RuntimeError as e:  # missing/unreachable database — a config error, not a chat error
        raise HTTPException(503, str(e)) from e

    async def events():
        # characters seen, so a gateway that reports no usage can still be billed
        chars = len(body.question) + sum(len(str(m.get("content", ""))) for m in body.history)
        try:
            async for event in chat_stream(body.question, body.history, llm, ctx):
                if isinstance(event, dict):
                    yield sse("status", event)
                else:
                    chars += len(event)
                    yield sse("token", {"text": event})
            yield sse("done", {})
        except AuthenticationError:
            log.exception("chat key rejected (401)")
            yield sse("error", {"code": "auth", "message": _AUTH_MESSAGE})
        except Exception:
            log.exception("answer failed")
            yield sse("error", {"message": "Sorry, I hit a snag answering that — please try again."})
        finally:
            await bill(user, usage, chars)

    return EventSourceResponse(events())


PREVIEW_ROWS = 200  # ponytail: head-only preview; paginate if a sheet needs scrolling past this


@app.get("/api/sheet")
async def sheet(request: Request, name: str, limit: int = 50):
    """First rows of one extracted sheet, for the source preview panel."""
    await signed_in(request)  # free (no model call), but still that user's own data
    df = await run_in_threadpool(load_frame, request.state.sid, name)
    if df is None:
        raise HTTPException(404, f"No sheet named '{name}' in this session.")
    head = df.head(min(limit, PREVIEW_ROWS))
    return {
        "columns": [str(c) for c in df.columns],
        "rows": json.loads(head.to_json(orient="records", date_format="iso")),
        "total": len(df),
    }


class ModelBody(BaseModel):
    model: str = LLM_MODEL
    baseUrl: str = LLM_BASE_URL
    key: str = ""


@app.post("/api/model")
async def model_status(request: Request, body: ModelBody):
    """Is the chat model itself reachable? A one-token probe — embeddings and the
    reranker are deliberately not consulted, so the indicator means what it says."""
    await signed_in(request)  # one token; not worth billing, but not open to the world
    chat_key, embed_key = _keys(body.key)
    if not chat_key:
        return {"online": False, "reason": "No API key — add one in settings."}
    try:
        # inside the try: building the clients can raise too (bad base URL, embed model
        # refused), and a 500 here reads as "the whole API is down" instead of "model off"
        llm, _ = resources(chat_key, body.baseUrl, body.model, embed_key)
        await llm.ainvoke("ping", max_tokens=1)
        return {"online": True, "reason": ""}
    except AuthenticationError:
        return {"online": False, "reason": "The API key was rejected."}
    except Exception as e:
        log.warning("model probe failed | model=%s | %s", body.model, e)
        return {"online": False, "reason": "The model didn't respond."}


class ChartBody(BaseModel):
    sheet: str
    question: str
    model: str = LLM_MODEL
    baseUrl: str = LLM_BASE_URL
    key: str = ""


@app.post("/api/chart")
async def chart(request: Request, body: ChartBody):
    """Natural-language chart request → a Vega-Lite spec with its data attached."""
    user = await with_credits(request)
    chat_key, embed_key = _keys(body.key)
    if not chat_key:
        raise HTTPException(400, "Add an API key to generate charts.")
    llm, _ = resources(chat_key, body.baseUrl, body.model, embed_key)
    llm, usage = metered(llm)
    try:
        return await run_in_threadpool(
            build_chart, request.state.sid, body.sheet, body.question, llm
        )
    except ValueError as e:  # no data, or the model produced something unusable
        raise HTTPException(422, str(e)) from e
    except AuthenticationError as e:
        raise HTTPException(401, _AUTH_MESSAGE) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        log.exception("chart failed")
        raise HTTPException(500, "Couldn't build that chart — try rephrasing it.") from e
    finally:
        await bill(user, usage, len(body.question))


class FeedbackBody(BaseModel):
    rating: str  # "up" | "down"
    question: str = ""
    answer: str = ""
    model: str = ""


@app.post("/api/feedback")
async def feedback(request: Request, body: FeedbackBody):
    """Record thumbs signal for later review. Does not change live answers."""

    def write():
        with engine().begin() as c:
            c.execute(
                text(
                    "INSERT INTO analytics.feedback (sid, rating, model, question, answer) "
                    "VALUES (:sid, :rating, :model, :question, :answer)"
                ),
                {"sid": request.state.sid, **body.model_dump()},
            )

    try:
        await run_in_threadpool(write)
    except Exception:
        log.exception("feedback log failed")  # never fail the user's click
    return {"ok": True}
