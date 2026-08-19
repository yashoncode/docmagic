"use client";

import { useEffect, useState } from "react";
import { AlertCircle, Loader2, Sparkles, X } from "lucide-react";
import TopBar from "@/components/topbar";
import DocPanel from "@/components/doc-panel";
import Chat from "@/components/chat";
import AnalysisView from "@/components/analysis-view";
import ExportPdf from "@/components/export-pdf";
import Login from "@/components/login";
import {
  askChat,
  checkModel,
  getConfig,
  getMe,
  sse,
  uploadFiles,
  INGEST_STEPS,
  STATUS,
  type Config,
  type IngestResult,
  type ModelStatus,
  type Msg,
  type User,
} from "@/lib/api";

/** Endpoint + key survive a refresh but not the tab — sessionStorage, never localStorage:
 *  a pasted key must not outlive the session on a shared machine. */
function remember(key: string, set: (v: string) => void) {
  return (v: string) => {
    sessionStorage.setItem(key, v);
    set(v);
  };
}

export default function Page() {
  const [config, setConfig] = useState<Config | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [user, setUser] = useState<User | null | undefined>(undefined); // undefined = checking

  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState(""); // server-configured; no picker
  const [baseUrl, setBaseUrl] = useState("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null); // null = checking

  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [ingest, setIngest] = useState<IngestResult | null>(null);
  const [hasDocs, setHasDocs] = useState(false);

  const [messages, setMessages] = useState<Msg[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setModel(c.defaultModel);
        setBaseUrl(sessionStorage.getItem("baseUrl") || c.baseUrl);
        setApiKey(sessionStorage.getItem("apiKey") ?? "");
      })
      .catch((e) => setFatal(e.message));
    getMe().then(setUser).catch(() => setUser(null));
  }, []);

  // probe the chat model itself; debounced so typing a key doesn't fire a call per keystroke
  useEffect(() => {
    if (!model || !user) return; // the probe costs a model call, so it needs a session
    const ctl = new AbortController();
    const t = setTimeout(() => {
      setModelStatus(null); // back to "checking…" only once the debounce settles
      checkModel({ model, baseUrl, key: apiKey }, ctl.signal)
        .then(setModelStatus)
        .catch(() => {});
    }, 500);
    return () => {
      clearTimeout(t);
      ctl.abort();
    };
  }, [model, baseUrl, apiKey, user]);

  async function analyse() {
    if (!config || files.length === 0) return;
    setUploading(true);
    setError(null);
    setIngest(null);
    setIngestStatus("Reading your documents…");
    setMessages([]); // old chat referred to old documents
    try {
      const res = await uploadFiles(files, { key: apiKey, model, baseUrl });
      for await (const { event, data } of sse(res)) {
        if (event === "ingested") {
          setHasDocs(data.chunks > 0);
        } else if (event === "node") {
          setIngestStatus(INGEST_STEPS[data.name] ?? "Working…");
        } else if (event === "result") {
          if (data.reason === "no_text") {
            setError(
              "I couldn't read any text from those files — I also tried transcribing them as scans and got nothing usable.",
            );
          } else if (data.reason === "no_key") {
            setError("Documents indexed. Add your API key to chat and get an analysis.");
          } else {
            setIngest(data as IngestResult);
          }
        } else if (event === "error") {
          setError(data.message);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
      setIngestStatus(null);
      getMe().then(setUser); // the run just spent tokens — show the new balance
    }
  }

  async function send(question: string) {
    if (busy) return;
    setBusy(true);
    setError(null);
    setStatus("Thinking…");
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    try {
      const res = await askChat({
        question, history, model, baseUrl, key: apiKey, webOn: true, hasDocs,
      });
      for await (const { event, data } of sse(res)) {
        if (event === "status") {
          const key = data.stage ?? data.tool ?? data.route;
          setStatus((key && STATUS[key]) || "Working…");
        } else if (event === "token") {
          setStatus(null);
          setMessages((m) => {
            const next = [...m];
            const last = next[next.length - 1];
            // models often emit leading whitespace before the real answer
            const content = last.content ? last.content + data.text : data.text.trimStart();
            next[next.length - 1] = { ...last, content };
            return next;
          });
        } else if (event === "error") {
          setError(data.message);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sorry, I hit a snag answering that.");
    } finally {
      setBusy(false);
      setStatus(null);
      getMe().then(setUser);
      // drop the placeholder if nothing streamed back
      setMessages((m) =>
        m.length && m[m.length - 1].role === "assistant" && !m[m.length - 1].content
          ? m.slice(0, -1)
          : m,
      );
    }
  }

  if (fatal) {
    return (
      <main className="grid min-h-dvh place-items-center p-6 text-center">
        <div>
          <AlertCircle className="mx-auto size-7 text-muted" aria-hidden />
          <p className="display mt-3 text-lg">{fatal}</p>
          <p className="meta mt-2 text-muted">uvicorn api:app --reload --port 8000</p>
        </div>
      </main>
    );
  }

  if (!config || user === undefined) {
    return (
      <main className="grid min-h-dvh place-items-center">
        <span className="meta pulse text-muted">loading…</span>
      </main>
    );
  }

  if (!user) return <Login config={config} onSignedIn={setUser} />;

  return (
    <div className="min-h-dvh">
      <TopBar
        config={config}
        apiKey={apiKey}
        setApiKey={remember("apiKey", setApiKey)}
        baseUrl={baseUrl}
        setBaseUrl={remember("baseUrl", setBaseUrl)}
        modelStatus={modelStatus}
        user={user}
        onSignedOut={() => {
          // the server just purged the uploads — don't leave their results on screen
          setUser(null);
          setIngest(null);
          setMessages([]);
          setFiles([]);
          setHasDocs(false);
        }}
      />

      {/* headline row — eyebrow + title left, live status + primary action right */}
      <div className="no-print shrink-0 px-4 pt-6 pb-4 sm:px-6">
        <div className="rise flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="min-w-0">
            <p className="eyebrow text-accent">Document intelligence</p>
            <h1 className="display mt-1.5 text-[26px] sm:text-[30px]">
              Ask your documents anything
            </h1>
          </div>

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <span
              className={`meta hidden items-center gap-1.5 md:inline-flex ${
                uploading ? "shimmer" : "text-muted"
              }`}
            >
              <span
                className={`size-1.5 rounded-full transition-colors ${
                  uploading ? "animate-ping bg-accent" : hasDocs ? "bg-success" : "bg-border-strong"
                }`}
              />
              {ingestStatus ?? (hasDocs ? "documents indexed" : "not indexed yet")}
            </span>
            <span className="meta hidden text-muted lg:inline">·</span>
            <span className="meta hidden text-muted lg:inline">
              {files.length}/{config.maxFiles} files
            </span>
            {ingest && <ExportPdf />}
            <button
              type="button"
              onClick={analyse}
              disabled={uploading || files.length === 0}
              className="btn btn-primary ml-1"
            >
              {uploading ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <Sparkles className="size-4" aria-hidden />
              )}
              {uploading ? "Analysing…" : "Analyse"}
            </button>
          </div>
        </div>
      </div>

      {!config.dbReady && (
        <div className="well rise no-print mx-4 mb-3 shrink-0 px-3 py-2.5 text-[13px] sm:mx-6">
          <p className="flex items-center gap-2 font-medium">
            <AlertCircle className="size-4 shrink-0 text-accent" aria-hidden />
            Setup needed — <code className="meta">POSTGRES_URL</code> isn&apos;t set
          </p>
          <p className="mt-1 pl-6 text-xs leading-relaxed text-muted">
            Upload and chat need Postgres with the <code className="meta">vector</code>{" "}
            extension. Grab a free database from{" "}
            <a
              href="https://supabase.com"
              target="_blank"
              rel="noreferrer"
              className="text-accent underline underline-offset-2"
            >
              Supabase
            </a>{" "}
            or{" "}
            <a
              href="https://neon.tech"
              target="_blank"
              rel="noreferrer"
              className="text-accent underline underline-offset-2"
            >
              Neon
            </a>
            , add it to <code className="meta">.env</code>, and restart the API. Your API
            keys are already configured.
          </p>
        </div>
      )}

      {error && (
        <div className="well rise no-print mx-4 mb-3 flex shrink-0 items-start gap-2 px-3 py-2.5 text-[13px] sm:mx-6">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden />
          <p className="flex-1">{error}</p>
          <button
            type="button"
            onClick={() => setError(null)}
            aria-label="Dismiss"
            className="text-muted hover:text-foreground"
          >
            <X className="size-4" aria-hidden />
          </button>
        </div>
      )}

      {/* Workspace: upload and conversation stay together; analysis follows below. */}
      <main className="space-y-4 px-4 pb-4 sm:px-6 sm:pb-6">
        {/* masthead for the exported PDF — on screen the topbar already says all this.
            The date only ever renders on the client, so the prerendered build date it
            replaces is a hydration mismatch by design. */}
        {ingest && (
          <header className="print-only mb-5 border-b border-border pb-3">
            <p className="eyebrow text-accent">DocMagic · document intelligence</p>
            <h2 className="display mt-1 text-lg">Document analysis</h2>
            <p className="meta mt-1.5 text-muted">
              {ingest.metadata.map((m) => m.source).join(" · ")}
            </p>
            <p className="meta mt-0.5 text-muted" suppressHydrationWarning>
              Generated {new Date().toLocaleString()}
            </p>
          </header>
        )}

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
          <DocPanel
            files={files}
            setFiles={setFiles}
            maxFiles={config.maxFiles}
            analysed={hasDocs}
            busy={uploading}
            sheets={ingest?.sheets ?? []}
          />

          <section className="panel no-print flex min-h-[560px] flex-col">
            <div className="panel-head">
              <span className="eyebrow text-muted">Conversation</span>
            </div>
            <Chat
              messages={messages}
              status={status}
              busy={busy}
              suggestions={
                ingest?.suggestions?.length ? ingest.suggestions : config.suggestions
              }
              model={model}
              onSend={send}
            />
          </section>
        </div>

        {ingest && (
          <AnalysisView ingest={ingest} model={model} baseUrl={baseUrl} apiKey={apiKey} />
        )}
      </main>
    </div>
  );
}
