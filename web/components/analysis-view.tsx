"use client";

import { BadgeCheck } from "lucide-react";
import Markdown from "react-markdown";
import VegaChart from "@/components/vega-chart";
import type { IngestResult } from "@/lib/api";

type Props = {
  ingest: IngestResult;
  model: string;
  baseUrl: string;
  apiKey: string;
};

/** Post-upload analysis: the reviewed summary beside the promptable charts. */
export default function AnalysisView({ ingest, model, baseUrl, apiKey }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <section data-print-section="summary" className="panel flex min-h-[360px] flex-col">
        <div className="panel-head">
          <span className="eyebrow text-muted">Summary</span>
          {ingest.review.approved && (
            <span className="chip chip-ok ml-auto">
              <BadgeCheck className="size-3" aria-hidden />
              verified
            </span>
          )}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">
          <div className="flex flex-wrap gap-1.5">
            {ingest.metadata.map((metadata) => (
              <span key={metadata.source} className="chip normal-case">
                <span className="font-medium text-foreground">{metadata.source}</span>
                <span className="opacity-60">·</span>
                {metadata.doc_type ?? metadata.kind}
                <span className="opacity-60">·</span>
                {metadata.kind === "PDF"
                  ? `${metadata.pages} pages`
                  : `${Object.keys(metadata.sheets ?? {}).length} sheets`}
              </span>
            ))}
          </div>
          {ingest.sheets.length > 0 && (
            <p className="meta mt-3 text-muted">queryable tables: {ingest.sheets.join(", ")}</p>
          )}
          <div className="md mt-5 text-[13.5px]">
            <Markdown>{ingest.summary}</Markdown>
          </div>
        </div>
      </section>

      <section data-print-section="charts" className="panel flex min-h-[360px] flex-col">
        <div className="panel-head">
          <span className="eyebrow text-muted">Charts</span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">
          <VegaChart
            sheets={ingest.sheets}
            hints={ingest.chartHints}
            model={model}
            baseUrl={baseUrl}
            apiKey={apiKey}
          />
        </div>
      </section>
    </div>
  );
}
