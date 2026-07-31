"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, Sheet, Upload, X } from "lucide-react";
import { getSheet, type SheetPreview } from "@/lib/api";

type Props = {
  files: File[];
  setFiles: (f: File[]) => void;
  maxFiles: number;
  analysed: boolean;
  busy: boolean;
  /** Extracted sheet names from the ingest result — previewable once analysed. */
  sheets: string[];
};

type Tab = { kind: "pdf"; label: string; file: File } | { kind: "sheet"; label: string };

export default function DocPanel({ files, setFiles, maxFiles, analysed, busy, sheets }: Props) {
  const picker = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [active, setActive] = useState(0);

  const tabs: Tab[] = [
    ...files
      .filter((f) => f.name.toLowerCase().endsWith(".pdf"))
      .map((f): Tab => ({ kind: "pdf", label: f.name, file: f })),
    ...sheets.map((s): Tab => ({ kind: "sheet", label: s })),
  ];
  const tab = tabs[Math.min(active, tabs.length - 1)];

  function accept(list: FileList | null) {
    if (!list) return;
    const ok = Array.from(list).filter((f) => /\.(pdf|xlsx|xlsm)$/i.test(f.name));
    if (ok.length) {
      setFiles(ok.slice(0, maxFiles));
      setActive(0);
    }
  }

  return (
    <section data-print-section="data" className="panel flex min-h-0 flex-col">
      <div className="panel-head">
        <span className="eyebrow text-muted">Source</span>
        {files.length > 0 && (
          <>
            <span className={`chip ml-1 ${analysed ? "chip-ok" : ""}`}>
              {busy ? "running" : analysed ? "indexed" : "ready"}
            </span>
            <button
              type="button"
              onClick={() => picker.current?.click()}
              className="btn btn-ghost no-print ml-auto px-2.5 py-1 text-xs"
            >
              <Upload className="size-3.5" aria-hidden />
              Replace
            </button>
          </>
        )}
      </div>

      <input
        ref={picker}
        type="file"
        multiple
        accept=".pdf,.xlsx,.xlsm"
        onChange={(e) => accept(e.target.files)}
        className="sr-only"
      />

      {files.length === 0 ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            accept(e.dataTransfer.files);
          }}
          className={`m-3 flex flex-1 flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center transition-colors ${
            over ? "border-accent bg-accent-soft" : "border-border bg-surface-2"
          }`}
        >
          <div className="grid size-11 place-items-center rounded-full bg-background ring-1 ring-border">
            <Upload className="size-[18px] text-muted" aria-hidden />
          </div>
          <p className="display mt-3 text-[15px]">Drop a document</p>
          <p className="mt-1 max-w-[26ch] text-xs leading-relaxed text-muted">
            PDF or spreadsheet, up to {maxFiles} files. Text is indexed; sheets become
            queryable tables.
          </p>
          <button
            type="button"
            onClick={() => picker.current?.click()}
            className="btn btn-ghost mt-4"
          >
            Choose files
          </button>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <ul className="shrink-0 space-y-1 p-3">
            {files.map((f) => (
              <li
                key={f.name}
                className="flex items-center gap-2 rounded-lg bg-surface px-2.5 py-2 text-[13px]"
              >
                {f.name.toLowerCase().endsWith(".pdf") ? (
                  <FileText className="size-4 shrink-0 text-muted" aria-hidden />
                ) : (
                  <Sheet className="size-4 shrink-0 text-muted" aria-hidden />
                )}
                <span className="truncate">{f.name}</span>
                <span className="meta ml-auto shrink-0 text-muted">
                  {(f.size / 1024).toFixed(0)} KB
                </span>
                <button
                  type="button"
                  onClick={() => setFiles(files.filter((x) => x !== f))}
                  aria-label={`Remove ${f.name}`}
                  className="shrink-0 text-muted hover:text-foreground"
                >
                  <X className="size-3.5" aria-hidden />
                </button>
              </li>
            ))}
          </ul>

          {tabs.length > 1 && (
            <div
              role="tablist"
              aria-label="Preview"
              className="flex shrink-0 gap-3 overflow-x-auto border-t border-border px-3"
            >
              {tabs.map((t, i) => (
                <button
                  key={`${t.kind}:${t.label}`}
                  type="button"
                  role="tab"
                  aria-selected={t === tab}
                  onClick={() => setActive(i)}
                  data-active={t === tab}
                  className="tab shrink-0 whitespace-nowrap"
                >
                  {t.label}
                </button>
              ))}
            </div>
          )}

          <div className="min-h-0 flex-1 border-t border-border bg-surface-2 p-3">
            {!tab ? (
              <Placeholder
                title="Spreadsheet ready"
                note="Analyse it to preview the extracted sheets."
              />
            ) : tab.kind === "pdf" ? (
              // keyed: a new tab remounts with fresh state, so no effect has to reset it
              <PdfPreview key={tab.label} file={tab.file} />
            ) : (
              <SheetPreviewTable key={tab.label} name={tab.label} />
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function Placeholder({ title, note }: { title: string; note: string }) {
  return (
    <div className="grid h-full min-h-[220px] place-items-center rounded-lg border border-border bg-background text-center">
      <div>
        <Sheet className="mx-auto size-6 text-muted" aria-hidden />
        <p className="display mt-2 text-[15px]">{title}</p>
        <p className="mt-1 text-xs text-muted">{note}</p>
      </div>
    </div>
  );
}

function PdfPreview({ file }: { file: File }) {
  const [url] = useState(() => URL.createObjectURL(file));
  useEffect(() => () => URL.revokeObjectURL(url), [url]); // object URLs leak until revoked

  return (
    <iframe
      src={url}
      title={`Preview of ${file.name}`}
      className="h-full min-h-[220px] w-full rounded-lg border border-border bg-background"
    />
  );
}

function SheetPreviewTable({ name }: { name: string }) {
  const [data, setData] = useState<SheetPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getSheet(name, controller.signal)
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e instanceof Error ? e.message : "Load failed.");
      });
    return () => controller.abort();
  }, [name]);

  if (error) return <Placeholder title="Preview unavailable" note={error} />;
  if (!data) {
    return (
      <div className="grid h-full min-h-[220px] place-items-center rounded-lg border border-border bg-background">
        <span className="meta pulse text-muted">loading {name}…</span>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-[220px] flex-col rounded-lg border border-border bg-background">
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-collapse text-[12px]">
          <thead className="sticky top-0 bg-surface">
            <tr>
              {data.columns.map((c) => (
                <th
                  key={c}
                  scope="col"
                  className="border-b border-border px-2.5 py-1.5 text-left font-medium whitespace-nowrap"
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, i) => (
              <tr key={i} className="odd:bg-surface-2">
                {data.columns.map((c) => (
                  <td key={c} className="border-b border-border px-2.5 py-1.5 whitespace-nowrap">
                    {row[c] === null || row[c] === undefined ? "" : String(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="meta shrink-0 border-t border-border px-2.5 py-1.5 text-muted">
        {data.rows.length} of {data.total} rows
      </p>
    </div>
  );
}
