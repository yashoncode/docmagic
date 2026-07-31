"use client";

import { Moon, Settings, Sun } from "lucide-react";
import type { Config } from "@/lib/api";

/** Flips `data-theme` on <html>; the palette and which icon shows are CSS's job, so
 *  there's no React state to keep in sync with the DOM (and nothing to mis-hydrate). */
function ThemeToggle() {
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
};

/** Brand row. Endpoint and key live in a settings popover; the model is the server's. */
export default function TopBar({ config, apiKey, setApiKey, baseUrl, setBaseUrl }: Props) {
  return (
    <header className="no-print flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5 sm:px-6">
      <span className="display text-[15px]">DocMagic</span>
      <span className="chip">XEON AI</span>
      {config.needsKey && !apiKey && <span className="chip chip-accent">key needed</span>}

      <span className="ml-auto" />
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
        className="panel fixed inset-auto top-14 right-4 m-0 w-[min(320px,calc(100vw-2rem))] space-y-3 p-4 shadow-lg"
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
            Kept in this tab only — never stored.
          </p>
        </div>
      </div>
    </header>
  );
}
