"use client";

import Link from "next/link";
import { LogOut, Moon, Settings, ShieldCheck, Sun } from "lucide-react";
import { logout, type Config, type ModelStatus, type User } from "@/lib/api";

/** Flips `data-theme` on <html>; the palette and which icon shows are CSS's job, so
 *  there's no React state to keep in sync with the DOM (and nothing to mis-hydrate).
 *  Exported because the login page has no topbar but still needs the switch. */
export function ThemeToggle() {
  function flip() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
  }

  return (
    <button
      type="button"
      onClick={flip}
      aria-label="Toggle dark mode"
      className="btn btn-ghost size-9 p-0"
    >
      <Sun size={20} className="theme-dark-only" aria-hidden />
      <Moon size={20} className="theme-light-only" aria-hidden />
    </button>
  );
}

type Props = {
  config: Config;
  apiKey: string;
  setApiKey: (v: string) => void;
  baseUrl: string;
  setBaseUrl: (v: string) => void;
  modelStatus: ModelStatus | null; // null while the probe is in flight
  user: User;
  onSignedOut: () => void;
};

/** Brand row. Endpoint and key live in a settings popover; the model is the server's. */
export default function TopBar({
  config, apiKey, setApiKey, baseUrl, setBaseUrl, modelStatus, user, onSignedOut,
}: Props) {
  const online = modelStatus?.online ?? false;
  const label = modelStatus ? (online ? "on" : "off") : "checking…";
  const low = user.tokens_left < config.freeTokens / 10;

  return (
    <header className="glass-bar no-print flex flex-wrap items-center gap-2 px-4 py-2.5 sm:px-6">
      <span className="display text-[15px]">DocMagic</span>
      <span className="chip">XEON AI</span>
      {config.needsKey && !apiKey && <span className="chip chip-accent">key needed</span>}

      <span className="ml-auto" />

      {/* model status: a live one-token probe of the chat model — not the key's presence,
          and not the embedding or reranker models */}
      <button type="button" popoverTarget="model-info" className="chip gap-1.5">
        <span
          aria-hidden
          className={`size-2 rounded-full ${
            !modelStatus ? "animate-pulse bg-border-strong" : online ? "bg-success" : "bg-muted"
          }`}
        />
        AI model {label}
      </button>
      <div
        id="model-info"
        popover="auto"
        className="panel fixed inset-auto top-14 right-4 m-0 max-w-[min(320px,calc(100vw-2rem))] p-3"
      >
        <p className="eyebrow text-muted">Model</p>
        <p className="meta mt-1 break-all">{config.defaultModel}</p>
        {modelStatus?.reason && (
          <p className="mt-1.5 text-xs text-muted">{modelStatus.reason}</p>
        )}
      </div>

      {/* token credits + account, in one popover */}
      <button type="button" popoverTarget="account" className="chip gap-1.5">
        <span
          aria-hidden
          className="grid size-4 place-items-center rounded-full bg-accent text-[9px] font-semibold text-background"
        >
          {(user.name || user.email).charAt(0).toUpperCase()}
        </span>
        <span className={low ? "text-accent" : undefined}>
          {user.tokens_left.toLocaleString()} tokens
        </span>
      </button>
      <div
        id="account"
        popover="auto"
        className="panel fixed inset-auto top-14 right-4 m-0 w-[min(300px,calc(100vw-2rem))] p-4"
      >
        <p className="eyebrow text-muted">Account</p>
        <p className="mt-1 truncate text-sm font-medium">{user.name}</p>
        <p className="meta truncate text-muted">{user.email}</p>
        <dl className="well mt-3 grid grid-cols-2 gap-1 px-3 py-2 text-[13px]">
          <dt className="text-muted">Tokens left</dt>
          <dd className="text-right font-medium">{user.tokens_left.toLocaleString()}</dd>
          <dt className="text-muted">Tokens used</dt>
          <dd className="text-right font-medium">{user.tokens_used.toLocaleString()}</dd>
        </dl>
        {low && (
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Running low — an admin can top you up.
          </p>
        )}
        <div className="mt-3 flex items-center gap-2">
          {user.is_admin && (
            <Link href="/admin" className="btn btn-ghost flex-1">
              <ShieldCheck className="size-4" aria-hidden />
              Users
            </Link>
          )}
          <button
            type="button"
            onClick={() => logout().then(onSignedOut)}
            className="btn btn-ghost flex-1"
          >
            <LogOut className="size-4" aria-hidden />
            Sign out
          </button>
        </div>
      </div>

      <ThemeToggle />

      {/* native popover: light-dismiss and Escape come for free */}
      <button
        type="button"
        popoverTarget="settings"
        aria-label="Settings"
        className="btn btn-ghost size-9 p-0"
      >
        <Settings size={20} aria-hidden />
      </button>
      <div
        id="settings"
        popover="auto"
        className="panel fixed inset-auto top-14 right-4 m-0 w-[min(320px,calc(100vw-2rem))] space-y-3 p-4"
      >
        <p className="eyebrow text-muted">Settings</p>
        <div>
          <label htmlFor="base-url" className="meta text-muted">
            API base URL
          </label>
          <input
            id="base-url"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder={config.baseUrl}
            className="field meta mt-1"
          />
        </div>
        <div>
          <label htmlFor="api-key" className="meta text-muted">
            API key
          </label>
          <input
            id="api-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={config.needsKey ? "paste your API key" : "using server key"}
            autoComplete="off"
            className="field meta mt-1"
          />
          <p className="mt-1.5 text-xs leading-relaxed text-muted">
            Kept in this browser tab — survives a refresh, gone when you close it.
          </p>
        </div>
      </div>
    </header>
  );
}
