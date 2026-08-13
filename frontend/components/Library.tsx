"use client";

import { useCallback, useRef, useState } from "react";
import { deleteDocument, uploadDocument } from "@/lib/api";
import type { DocumentSummary, Health } from "@/lib/types";

export function Library({
  documents,
  health,
  onChanged,
}: {
  documents: DocumentSummary[];
  health: Health | null;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      setBusy(true);
      setError(null);
      setNotice(null);
      const messages: string[] = [];
      for (const file of Array.from(files)) {
        try {
          const result = (await uploadDocument(file)) as {
            n_chunks: number;
            skipped: boolean;
            replaced: boolean;
          };
          messages.push(
            result.skipped
              ? `${file.name} — already indexed, unchanged`
              : `${file.name} — ${result.n_chunks} chunks${result.replaced ? " (replaced)" : ""}`,
          );
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
      if (messages.length) setNotice(messages.join(" · "));
      setBusy(false);
      onChanged();
    },
    [onChanged],
  );

  const remove = async (id: string) => {
    try {
      await deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <aside className="flex w-full flex-col gap-4 border-ink-200 lg:h-screen lg:w-80 lg:shrink-0 lg:overflow-y-auto lg:border-r lg:p-5 dark:border-ink-800">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">⚓ Anchor</h1>
        <p className="mt-0.5 text-xs text-ink-400">
          Hybrid retrieval · grounded citations
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) void upload(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-5 text-center transition ${
          dragging
            ? "border-anchor-500 bg-anchor-500/10"
            : "border-ink-200 hover:border-anchor-400 dark:border-ink-800"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".txt,.md,.markdown,.pdf,.rst,.csv,.json"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) void upload(e.target.files);
            e.target.value = "";
          }}
        />
        <p className="text-sm font-medium">
          {busy ? "Indexing…" : "Drop files or click to upload"}
        </p>
        <p className="mt-1 text-[11px] text-ink-400">
          .md · .txt · .pdf · .csv · .json
        </p>
      </div>

      {error && (
        <p className="rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
      {notice && (
        <p className="rounded-lg bg-anchor-500/10 px-3 py-2 text-xs text-anchor-600 dark:text-anchor-400">
          {notice}
        </p>
      )}

      <div className="flex-1">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
          Library ({documents.length})
        </p>
        {documents.length === 0 ? (
          <p className="text-xs text-ink-400">
            Nothing indexed yet. Try{" "}
            <code className="rounded bg-ink-100 px-1 dark:bg-ink-800">
              data/samples/
            </code>
            .
          </p>
        ) : (
          <ul className="space-y-1.5">
            {documents.map((doc) => (
              <li
                key={doc.id}
                className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-ink-100 dark:hover:bg-ink-800/60"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm">{doc.title}</p>
                  <p className="text-[11px] text-ink-400">
                    {doc.n_chunks} chunks · {(doc.n_chars / 1000).toFixed(1)}k chars
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => remove(doc.id)}
                  aria-label={`Remove ${doc.title}`}
                  className="opacity-0 transition group-hover:opacity-100 hover:text-red-500"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {health && (
        <div className="space-y-1 border-t border-ink-200 pt-3 text-[11px] text-ink-400 dark:border-ink-800">
          <p>
            <span className="font-medium">{health.chunks}</span> chunks ·{" "}
            {health.embeddings === "ready" ? (
              <span className="text-anchor-600 dark:text-anchor-400">
                hybrid retrieval
              </span>
            ) : (
              <span className="text-amber-600 dark:text-amber-400">
                BM25 only
              </span>
            )}
          </p>
          <p className="truncate">
            {health.local_generation ? (
              <span className="text-anchor-600 dark:text-anchor-400">
                ⬤ local
              </span>
            ) : (
              <span>☁ {health.provider}</span>
            )}{" "}
            · {health.model}
          </p>
          {health.notes.map((note) => (
            <p key={note} className="text-amber-600 dark:text-amber-400">
              {note}
            </p>
          ))}
        </div>
      )}
    </aside>
  );
}
