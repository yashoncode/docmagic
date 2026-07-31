"use client";

import { useState } from "react";
import { Download } from "lucide-react";

/** Sections the print stylesheet can include; the key matches `data-print-section`. */
const SECTIONS = [
  { key: "data", label: "Document data (preview table)" },
  { key: "summary", label: "Summary" },
  { key: "charts", label: "Charts (including pinned)" },
] as const;

const REDRAW_MS = 500; // ponytail: fixed wait for charts to re-embed in the light palette

/** Export via the browser's own print-to-PDF — nothing to bundle, and it paginates,
 *  picks the page size and embeds fonts better than any canvas-to-PDF library would. */
export default function ExportPdf() {
  const [picked, setPicked] = useState<Record<string, boolean>>({
    data: true,
    summary: true,
    charts: true,
  });
  const chosen = SECTIONS.filter((s) => picked[s.key]);

  async function save() {
    const root = document.documentElement;
    document.getElementById("export")?.hidePopover(); // else the picker itself prints

    for (const s of SECTIONS) root.classList.toggle(`print-${s.key}`, !!picked[s.key]);

    // charts are canvas: Vega bakes the palette in at embed time, so dark-mode text
    // would print invisible on white. Flip to light and let them redraw first.
    const theme = root.dataset.theme;
    if (theme === "dark") {
      root.dataset.theme = "light";
      await new Promise((done) => setTimeout(done, REDRAW_MS));
    }

    const restore = () => {
      if (theme) root.dataset.theme = theme;
      for (const s of SECTIONS) root.classList.remove(`print-${s.key}`);
      window.removeEventListener("afterprint", restore);
    };
    window.addEventListener("afterprint", restore);
    window.print();
  }

  return (
    <>
      <button type="button" popoverTarget="export" className="btn btn-ghost">
        <Download className="size-4" aria-hidden />
        Export
      </button>
      <div
        id="export"
        popover="auto"
        className="panel fixed inset-auto top-32 right-4 m-0 w-[min(320px,calc(100vw-2rem))] space-y-3 p-4 shadow-lg"
      >
        <p className="eyebrow text-muted">Include in the PDF</p>
        {SECTIONS.map((s) => (
          <label key={s.key} className="flex items-center gap-2 text-[13px]">
            <input
              type="checkbox"
              checked={!!picked[s.key]}
              onChange={(e) => setPicked((p) => ({ ...p, [s.key]: e.target.checked }))}
              className="size-4 accent-accent"
            />
            {s.label}
          </label>
        ))}
        <button
          type="button"
          onClick={save}
          disabled={chosen.length === 0}
          className="btn btn-primary w-full"
        >
          Save as PDF
        </button>
        <p className="text-xs leading-relaxed text-muted">
          Opens your print dialog — choose &ldquo;Save as PDF&rdquo; as the destination.
        </p>
      </div>
    </>
  );
}
