/** Thin client for the FastAPI backend. `credentials: include` everywhere — the
 *  session id lives in an httpOnly cookie the API sets. */

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Config = {
  defaultModel: string;
  baseUrl: string;
  needsKey: boolean;
  dbReady: boolean;
  maxFiles: number;
  suggestions: string[];
  /** Empty when the server has no GOOGLE_CLIENT_ID — Google sign-in is then hidden. */
  googleClientId: string;
  freeTokens: number;
  minPassword: number;
};

export type User = {
  id: number;
  email: string;
  name: string;
  tokens_left: number;
  tokens_used: number;
  is_admin: boolean;
  created?: string;
  last_seen?: string;
};

/** A Vega-Lite spec. Opaque here — vega-embed validates it at render time, and
 *  the server already stripped anything that could fetch a remote resource. */
export type TopLevelSpec = Record<string, unknown>;

export type SheetInfo = { rows: number; columns: string[] };
export type DocMeta = {
  source: string;
  kind: "PDF" | "Excel";
  doc_type?: string;
  pages?: number;
  sheets?: Record<string, SheetInfo>;
};
export type IngestResult = {
  summary: string;
  metadata: DocMeta[];
  review: { approved?: boolean; notes?: string };
  sheets: string[];
  chartHints: Record<string, string[]>;
  /** Document-aware chat starters; empty when the model couldn't produce any. */
  suggestions: string[];
};

export type Msg = { role: "user" | "assistant"; content: string };

/** Every frame either endpoint can emit. Discriminating on `event` in a
 *  for-await narrows `data` for free at the call site. */
export type SseEvent =
  | { event: "ingested"; data: { chunks: number } }
  | { event: "node"; data: { name: string } }
  | { event: "result"; data: Partial<IngestResult> & { reason?: string } }
  | { event: "status"; data: { stage?: string; route?: string; tool?: string } }
  | { event: "token"; data: { text: string } }
  | { event: "error"; data: { message: string; code?: string } }
  | { event: "done"; data: Record<string, never> };

/** Parse an SSE body. EventSource can't POST, so we read the stream ourselves. */
export async function* sse(res: Response): AsyncGenerator<SseEvent> {
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // frames are blank-line separated; sse-starlette emits CRLF
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        let event = "message";
        let data = "";
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
          // anything else (": ping" keep-alives) is a comment
        }
        // the server is the only producer of these frames — one cast, here
        if (data) yield { event, data: JSON.parse(data) } as SseEvent;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function detail(res: Response, fallback: string) {
  try {
    return (await res.json()).detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function getConfig(): Promise<Config> {
  const res = await fetch(`${API}/api/config`, { credentials: "include" });
  if (!res.ok) throw new Error("Could not reach the DocMagic API.");
  return res.json();
}

/** The signed-in user, or null. Never throws for "not signed in". */
export async function getMe(): Promise<User | null> {
  const res = await fetch(`${API}/api/me`, { credentials: "include" });
  if (!res.ok) return null;
  return (await res.json()).user;
}

/** Exchange Google's ID token for a session cookie. */
export async function googleLogin(credential: string): Promise<User> {
  const res = await fetch(`${API}/api/auth/google`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ credential }),
  });
  if (!res.ok) throw new Error(await detail(res, "Sign-in failed."));
  return res.json();
}

/** Email sign-in or sign-up — same shape, different path. */
export async function emailAuth(
  mode: "login" | "signup",
  body: { email: string; password: string; name?: string },
): Promise<User> {
  const res = await fetch(`${API}/api/auth/${mode}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await detail(res, "Sign-in failed."));
  return res.json();
}

export async function logout() {
  await fetch(`${API}/api/auth/logout`, { method: "POST", credentials: "include" });
}

export async function listUsers(): Promise<User[]> {
  const res = await fetch(`${API}/api/admin/users`, { credentials: "include" });
  if (!res.ok) throw new Error(await detail(res, "Could not load users."));
  return (await res.json()).users;
}

export async function setUserTokens(id: number, tokens: number): Promise<User> {
  const res = await fetch(`${API}/api/admin/users/${id}/tokens`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tokens }),
  });
  if (!res.ok) throw new Error(await detail(res, "Could not update credits."));
  return res.json();
}

export async function uploadFiles(
  files: File[],
  opts: { key: string; model: string; baseUrl: string },
): Promise<Response> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("key", opts.key);
  form.append("model", opts.model);
  form.append("base_url", opts.baseUrl);
  const res = await fetch(`${API}/api/upload`, {
    method: "POST",
    credentials: "include",
    body: form,
  });
  if (!res.ok) throw new Error(await detail(res, "Upload failed."));
  return res;
}

export async function askChat(body: {
  question: string;
  history: Msg[];
  model: string;
  baseUrl: string;
  key: string;
  webOn: boolean;
  hasDocs: boolean;
}): Promise<Response> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await detail(res, "Chat failed."));
  return res;
}

export type ModelStatus = { online: boolean; reason: string };

/** Probes the chat model itself (not embeddings or the reranker). */
export async function checkModel(
  body: { model: string; baseUrl: string; key: string },
  signal?: AbortSignal,
): Promise<ModelStatus> {
  const res = await fetch(`${API}/api/model`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  // the status is the whole diagnosis — 401 is a dead session, 5xx is the API itself
  if (!res.ok) return { online: false, reason: `The API answered ${res.status}.` };
  return res.json();
}

export type SheetPreview = {
  columns: string[];
  rows: Record<string, unknown>[];
  total: number;
};

/** First rows of one extracted sheet — only available after a successful analyse. */
export async function getSheet(name: string, signal?: AbortSignal): Promise<SheetPreview> {
  const res = await fetch(`${API}/api/sheet?name=${encodeURIComponent(name)}`, {
    credentials: "include",
    signal,
  });
  if (!res.ok) throw new Error(await detail(res, "Could not load that sheet."));
  return res.json();
}

export function sendFeedback(body: {
  rating: "up" | "down";
  question: string;
  answer: string;
  model: string;
}) {
  // fire and forget — a failed thumbs must never interrupt the user
  fetch(`${API}/api/feedback`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => {});
}

/** Pipeline stage / router lane / tool name → what the user sees while it happens.
 *  Each key is emitted by the step it names, so the line tracks real work. */
export const STATUS: Record<string, string> = {
  // stages, in the order a turn produces them
  routing: "Reading your question…",
  embedding: "Embedding the question, searching vectors…",
  keyword: "Keyword-matching exact terms…",
  rerank: "Re-ranking the best passages…",
  reading_sheet: "Loading the sheet…",
  composing: "Got the context — writing the answer…",
  // router lanes
  documents: "Asking the document expert…",
  data: "Asking the data analyst…",
  quant: "Asking the quant…",
  web: "Asking the web researcher…",
  // tool calls
  search_documents: "Searching your documents…",
  search_web: "Searching the web…",
  calculate: "Calculating…",
  describe_table: "Reading the spreadsheet…",
};

/** Upload stage / ingestion-graph node → live label. */
export const INGEST_STEPS: Record<string, string> = {
  clearing: "Clearing the previous upload…",
  indexing: "Splitting and embedding chunks…",
  tables: "Extracting spreadsheet tables…",
  extract_metadata: "Reading metadata…",
  analyst: "Analysing…",
  reviewer: "Reviewing for accuracy…",
  chart_hints: "Suggesting charts for your sheets…",
  questions: "Suggesting questions to ask…",
};
