import type {
  DocumentSummary,
  Health,
  SourceCard,
  StreamEvent,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<Health> {
  return json<Health>(await fetch(`${API_BASE}/api/health`, { cache: "no-store" }));
}

export async function fetchDocuments(): Promise<{
  documents: DocumentSummary[];
  total_chunks: number;
}> {
  return json(await fetch(`${API_BASE}/api/documents`, { cache: "no-store" }));
}

export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return json(
    await fetch(`${API_BASE}/api/documents`, { method: "POST", body: form }),
  );
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/documents/${id}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 204) {
    throw new Error(`Failed to delete document (${response.status})`);
  }
}

export async function search(q: string): Promise<SourceCard[]> {
  const params = new URLSearchParams({ q });
  const data = await json<{ results: SourceCard[] }>(
    await fetch(`${API_BASE}/api/search?${params}`, { cache: "no-store" }),
  );
  return data.results;
}

/**
 * Stream a chat answer.
 *
 * Written against fetch + ReadableStream rather than EventSource because
 * EventSource cannot issue a POST, and the question plus conversation history
 * belongs in a body, not a query string.
 */
export async function* streamChat(
  question: string,
  history: { role: "user" | "assistant"; content: string }[],
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep the status line */
    }
    yield { type: "error", message: detail };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. Keep the trailing partial
    // frame in the buffer — a chunk boundary can land mid-frame.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          yield JSON.parse(line.slice(6)) as StreamEvent;
        } catch {
          /* ignore a malformed frame rather than killing the stream */
        }
      }
    }
  }
}
