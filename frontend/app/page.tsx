"use client";

import { useCallback, useEffect, useState } from "react";
import { Chat } from "@/components/Chat";
import { Library } from "@/components/Library";
import { fetchDocuments, fetchHealth } from "@/lib/api";
import type { DocumentSummary, Health } from "@/lib/types";

export default function Page() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [docs, status] = await Promise.all([
        fetchDocuments(),
        fetchHealth(),
      ]);
      setDocuments(docs.documents);
      setHealth(status);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <Library documents={documents} health={health} onChanged={refresh} />
      {offline ? (
        <main className="flex flex-1 items-center justify-center p-10 text-center">
          <div>
            <p className="text-sm font-medium">Backend unreachable</p>
            <p className="mt-1 max-w-sm text-xs text-ink-400">
              Start it with{" "}
              <code className="rounded bg-ink-100 px-1 dark:bg-ink-800">
                uvicorn app.main:app --reload
              </code>{" "}
              from the <code>backend/</code> directory.
            </p>
            <button
              type="button"
              onClick={() => void refresh()}
              className="mt-4 rounded-lg border border-ink-200 px-3 py-1.5 text-xs dark:border-ink-800"
            >
              Retry
            </button>
          </div>
        </main>
      ) : (
        <Chat hasDocuments={documents.length > 0} />
      )}
    </div>
  );
}
