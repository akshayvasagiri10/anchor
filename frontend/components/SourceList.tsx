"use client";

import { useState } from "react";
import type { SourceCard } from "@/lib/types";

const BADGE: Record<string, string> = {
  keyword:
    "bg-amber-500/15 text-amber-700 dark:text-amber-400 ring-amber-500/25",
  semantic:
    "bg-sky-500/15 text-sky-700 dark:text-sky-400 ring-sky-500/25",
};

export function SourceList({
  sources,
  messageId,
}: {
  sources: SourceCard[];
  messageId: string;
}) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  if (sources.length === 0) return null;

  const toggle = (id: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="mt-4">
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
        {sources.length} source{sources.length === 1 ? "" : "s"} retrieved
      </p>
      <ol className="space-y-2">
        {sources.map((source) => {
          const isOpen = expanded.has(source.id);
          const preview =
            source.text.length > 180 && !isOpen
              ? `${source.text.slice(0, 180).trimEnd()}…`
              : source.text;

          return (
            <li
              key={source.chunk_id}
              id={`source-${messageId}-${source.id}`}
              className="scroll-mt-24 rounded-lg border border-ink-200 bg-white/70 p-3 transition dark:border-ink-800 dark:bg-ink-800/40"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 flex h-5 min-w-5 items-center justify-center rounded-md bg-anchor-500/15 px-1.5 text-[11px] font-semibold text-anchor-600 dark:text-anchor-400">
                  {source.id}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="truncate text-sm font-medium">
                      {source.document_title}
                    </span>
                    <span className="text-[11px] text-ink-400">
                      chunk {source.ordinal}
                    </span>
                    {source.matched_by.map((how) => (
                      <span
                        key={how}
                        className={`rounded px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset ${BADGE[how] ?? ""}`}
                        title={
                          how === "keyword"
                            ? "Found by BM25 lexical search"
                            : "Found by dense vector similarity"
                        }
                      >
                        {how}
                      </span>
                    ))}
                  </div>
                  <p className="mt-1.5 text-sm leading-relaxed text-ink-600 dark:text-ink-200">
                    {preview}
                  </p>
                  {source.text.length > 180 && (
                    <button
                      type="button"
                      onClick={() => toggle(source.id)}
                      className="mt-1 text-xs font-medium text-anchor-600 hover:underline dark:text-anchor-400"
                    >
                      {isOpen ? "Show less" : "Show full chunk"}
                    </button>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
