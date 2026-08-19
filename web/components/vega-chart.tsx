"use client";

import { useEffect, useRef, useState } from "react";
import { BarChart3, Loader2, Maximize2, Pin, PinOff, Sparkles } from "lucide-react";
import { API, type TopLevelSpec } from "@/lib/api";

type Props = {
  sheets: string[];
  /** Per-sheet suggestion chips, generated server-side from that sheet's real columns. */
  hints: Record<string, string[]>;
  model: string;
  baseUrl: string;
  apiKey: string;
  /** The model probe came back off — charting would only fail at the server. */
  offline: boolean;
};

type Chart = { id: number; spec: TopLevelSpec; note: string | null; title: string };

/** Vega config built from the app's CSS variables, so charts follow the theme toggle.
 *  Vega bakes colors in at embed time and its own "dark" theme ignores our palette. */
function themeConfig() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  const [text, muted, border, accent] = [
    v("--foreground"),
    v("--muted"),
    v("--border"),
    v("--accent"),
  ];
  return {
    background: "transparent",
    /* fit-x, not fit: `fit` squeezes the height too, and a step-sized discrete axis
       (5 rows ≈ 100px) then has to fit its own axis labels inside that — which
       collapses the marks to a sliver. Width still snaps to the panel; height grows. */
    autosize: { type: "fit-x", contains: "padding" } as const,
    font: "inherit",
    title: { color: text, subtitleColor: muted },
    axis: {
      labelColor: muted,
      titleColor: text,
      gridColor: border,
      domainColor: border,
      tickColor: border,
    },
    legend: { labelColor: muted, titleColor: text },
    header: { labelColor: muted, titleColor: text },
    mark: { color: accent },
    range: { category: [accent, "#17994f", "#e2a03f", "#8b5cf6", "#e05252", "#0ea5a5"] },
    view: { stroke: border },
  };
}

/** One embedded chart. Re-embeds on theme change and on fullscreen, both of which
 *  change values Vega only reads while embedding (colors, container width). */
function ChartView({ spec }: { spec: TopLevelSpec }) {
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [redraws, setRedraws] = useState(0);

  useEffect(() => {
    const bump = () => setRedraws((n) => n + 1);
    document.addEventListener("fullscreenchange", bump);
    const themes = new MutationObserver(bump);
    themes.observe(document.documentElement, { attributeFilter: ["data-theme"] });
    return () => {
      document.removeEventListener("fullscreenchange", bump);
      themes.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!host.current) return;
    let view: { finalize: () => void } | null = null;
    let cancelled = false;

    import("vega-embed").then(({ default: embed }) => {
      if (cancelled || !host.current) return;
      embed(host.current, spec, {
        mode: "vega-lite",
        actions: { export: true, source: false, compiled: false, editor: false },
        renderer: "canvas",
        // vega-embed reserves 38px on the right for the actions menu; leave it that
        // room or every chart overflows its container horizontally
        width: Math.max(240, host.current.clientWidth - 44),
        config: themeConfig(),
      })
        .then((result) => {
          view = result.view;
        })
        .catch((cause) =>
          setError(cause instanceof Error ? cause.message : "Could not render that chart."),
        );
    });

    return () => {
      cancelled = true;
      view?.finalize();
    };
  }, [spec, redraws]);

  if (error) return <p className="text-[13px] text-muted">{error}</p>;
  return <div ref={host} className="overflow-x-auto" />;
}

/** A chart in its own frame, with fullscreen and whatever pin control the caller gives. */
function ChartFrame({
  chart,
  children,
}: {
  chart: Chart;
  children: React.ReactNode;
}) {
  const frame = useRef<HTMLDivElement>(null);

  return (
    <div ref={frame} className="well rise p-3">
      <div className="flex items-center gap-1.5">
        <span className="meta truncate text-muted">{chart.title}</span>
        <div className="no-print ml-auto flex gap-1">
          {children}
          <button
            type="button"
            onClick={() =>
              document.fullscreenElement
                ? document.exitFullscreen()
                : frame.current?.requestFullscreen()
            }
            aria-label="Toggle fullscreen chart"
            title="Fullscreen (Esc to exit)"
            className="btn btn-ghost size-8 p-0"
          >
            <Maximize2 size={18} aria-hidden />
          </button>
        </div>
      </div>
      <ChartView spec={chart.spec} />
      {chart.note && <p className="meta mt-2 text-right text-muted">{chart.note}</p>}
    </div>
  );
}

export default function VegaChart({ sheets, hints, model, baseUrl, apiKey, offline }: Props) {
  const [picked, setPicked] = useState<string | null>(null);
  const sheet = picked && sheets.includes(picked) ? picked : (sheets[0] ?? "");
  const sheetHints = hints[sheet] ?? [];
  const [question, setQuestion] = useState("");
  const [chart, setChart] = useState<Chart | null>(null);
  const [pinned, setPinned] = useState<Chart[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(candidate: string) {
    const text = candidate.trim();
    if (!text || busy || !sheet) return;
    if (offline) {
      setError("The AI model is off — charts need it. Check the base URL and API key in Settings.");
      return;
    }
    setBusy(true);
    setError(null);
    setChart(null);
    try {
      const response = await fetch(`${API}/api/chart`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sheet, question: text, model, baseUrl, key: apiKey }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? "Could not build that chart.");
      }
      const result = await response.json();
      setChart({
        id: Date.now(),
        spec: result.spec,
        note: result.truncated ? `showing the first ${result.rows} rows` : `${result.rows} rows`,
        title: `${sheet} — ${text}`,
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not build that chart.");
    } finally {
      setBusy(false);
    }
  }

  if (!sheets.length) {
    return (
      <div className="well pop grid min-h-[200px] place-items-center text-center">
        <div>
          <BarChart3 className="mx-auto size-6 text-muted" aria-hidden />
          <p className="display mt-2 text-[15px]">No spreadsheet data</p>
          <p className="mt-1 text-xs text-muted">
            Upload an Excel file to create promptable charts.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          ask(question);
        }}
        className="no-print flex flex-wrap items-center gap-2"
      >
        {sheets.length > 1 && (
          <select
            value={sheet}
            onChange={(event) => setPicked(event.target.value)}
            aria-label="Sheet to chart"
            className="field meta w-auto"
          >
            {sheets.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        )}
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Describe a chart using this sheet's columns"
          aria-label="Describe a chart"
          className="field min-w-[200px] flex-1"
        />
        <button
          type="submit"
          disabled={busy || offline || !question.trim()}
          className="btn btn-primary"
        >
          {busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Sparkles className="size-4" aria-hidden />}
          Chart it
        </button>
      </form>

      {sheetHints.length > 0 && (
        <div className="stagger no-print mt-2 flex flex-wrap gap-1.5" aria-label="Chart ideas from uploaded columns">
          {sheetHints.map((hint) => (
            <button
              key={hint}
              type="button"
              onClick={() => {
                setQuestion(hint);
                ask(hint);
              }}
              className="chip normal-case hover:border-border-strong"
            >
              {hint}
            </button>
          ))}
        </div>
      )}

      {error && <p className="well rise mt-3 px-3 py-2 text-[13px]">{error}</p>}

      {busy && !chart && (
        <p className="meta shimmer mt-4 py-10 text-center">designing the chart...</p>
      )}

      {chart && (
        <div className="mt-4">
          <ChartFrame chart={chart}>
            <button
              type="button"
              onClick={() => {
                setPinned((p) => [chart, ...p]); // pinned charts survive the next request
                setChart(null);
              }}
              aria-label="Pin this chart"
              title="Pin — keeps it when you chart something else"
              className="btn btn-ghost size-8 p-0"
            >
              <Pin size={18} aria-hidden />
            </button>
          </ChartFrame>
        </div>
      )}

      {pinned.length > 0 && (
        <div className="mt-4 space-y-3">
          <p className="eyebrow text-muted">Pinned</p>
          {pinned.map((p) => (
            <ChartFrame key={p.id} chart={p}>
              <button
                type="button"
                onClick={() => setPinned((all) => all.filter((c) => c.id !== p.id))}
                aria-label="Unpin this chart"
                title="Unpin"
                className="btn btn-ghost size-8 p-0"
              >
                <PinOff size={18} aria-hidden />
              </button>
            </ChartFrame>
          ))}
        </div>
      )}
    </div>
  );
}
