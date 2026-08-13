"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import type { Message } from "@/lib/types";
import { AnswerText } from "./AnswerText";
import { SourceList } from "./SourceList";

const SUGGESTIONS = [
  "What is the refund window?",
  "Can I ship to a PO box?",
  "What does error ERR_4417 mean?",
];

const STAGE_LABEL: Record<string, string> = {
  retrieving: "Searching your documents…",
  thinking: "Thinking…",
  writing: "Writing…",
};

let counter = 0;
const nextId = () => `m${++counter}`;

export function Chat({ hasDocuments }: { hasDocuments: boolean }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const jumpToSource = useCallback((messageId: string, citation: number) => {
    const el = document.getElementById(`source-${messageId}-${citation}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("source-flash");
    // Force a reflow so the animation restarts on a repeat click.
    void el.offsetWidth;
    el.classList.add("source-flash");
  }, []);

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || streaming) return;

      const history = messages
        .filter((m) => !m.error)
        .map((m) => ({ role: m.role, content: m.content }));

      const userMessage: Message = {
        id: nextId(),
        role: "user",
        content: trimmed,
      };
      const answerId = nextId();
      const answer: Message = {
        id: answerId,
        role: "assistant",
        content: "",
        sources: [],
        stage: "retrieving",
      };

      setMessages((prev) => [...prev, userMessage, answer]);
      setInput("");
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const patch = (update: Partial<Message>) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === answerId ? { ...m, ...update } : m)),
        );

      try {
        for await (const event of streamChat(
          trimmed,
          history,
          controller.signal,
        )) {
          switch (event.type) {
            case "sources":
              patch({ sources: event.sources });
              break;
            case "status":
              patch({ stage: event.stage });
              break;
            case "token":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === answerId
                    ? { ...m, content: m.content + event.text, stage: "writing" }
                    : m,
                ),
              );
              break;
            case "done":
              patch({
                stage: "done",
                usage: event.usage,
                model: event.model,
                invalidCitations: event.invalid_citations ?? [],
              });
              break;
            case "error":
              patch({ stage: "done", error: event.message });
              break;
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          patch({
            stage: "done",
            error: err instanceof Error ? err.message : String(err),
          });
        } else {
          patch({ stage: "done" });
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [messages, streaming],
  );

  return (
    <main className="flex min-h-screen flex-1 flex-col">
      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="mx-auto w-full max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="pt-10 text-center">
              <h2 className="text-2xl font-semibold tracking-tight">
                Ask your documents
              </h2>
              <p className="mx-auto mt-2 max-w-md text-sm text-ink-400">
                Anchor searches by keyword and by meaning, then answers only
                from what it found — with a citation on every claim.
              </p>
              {hasDocuments && (
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => void send(s)}
                      className="rounded-full border border-ink-200 px-3 py-1.5 text-xs transition hover:border-anchor-400 hover:text-anchor-600 dark:border-ink-800 dark:hover:text-anchor-400"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <div key={message.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-anchor-500 px-4 py-2.5 text-sm text-white">
                  {message.content}
                </div>
              </div>
            ) : (
              <div key={message.id} className="max-w-full">
                {message.stage && message.stage !== "done" && !message.content && (
                  <p className="animate-pulse text-sm text-ink-400">
                    {STAGE_LABEL[message.stage]}
                  </p>
                )}

                {message.content && (
                  <div
                    className={`whitespace-pre-wrap text-[15px] leading-relaxed ${
                      streaming && message.stage === "writing"
                        ? "streaming-caret"
                        : ""
                    }`}
                  >
                    <AnswerText
                      text={message.content}
                      maxCitation={message.sources?.length ?? 0}
                      onCite={(n) => jumpToSource(message.id, n)}
                    />
                  </div>
                )}

                {message.error && (
                  <p className="rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-600 dark:text-red-400">
                    {message.error}
                  </p>
                )}

                <SourceList
                  sources={message.sources ?? []}
                  messageId={message.id}
                />

                {/* A model citing a source that doesn't exist is the exact
                    failure this project is about. Say it out loud rather
                    than letting a confident answer pass unchallenged. */}
                {message.invalidCitations &&
                  message.invalidCitations.length > 0 && (
                    <p className="mt-3 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
                      ⚠ This answer cited{" "}
                      {message.invalidCitations.map((n) => `[${n}]`).join(", ")},
                      which {message.invalidCitations.length === 1 ? "does" : "do"}{" "}
                      not exist — only {message.sources?.length ?? 0} source
                      {(message.sources?.length ?? 0) === 1 ? " was" : "s were"}{" "}
                      provided. Treat the surrounding claim as unverified.
                    </p>
                  )}

                {message.usage && (
                  <p className="mt-3 text-[11px] text-ink-400">
                    {message.model && <span>{message.model} · </span>}
                    {message.usage.input_tokens > 0 ||
                    message.usage.output_tokens > 0 ? (
                      <>
                        {message.usage.input_tokens.toLocaleString()} in ·{" "}
                        {message.usage.output_tokens.toLocaleString()} out
                        {message.usage.cache_read_input_tokens > 0 &&
                          ` · ${message.usage.cache_read_input_tokens.toLocaleString()} cached`}
                      </>
                    ) : (
                      <span>token counts not reported by this provider</span>
                    )}
                  </p>
                )}
              </div>
            ),
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-ink-200 px-5 py-4 dark:border-ink-800">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void send(input);
          }}
          className="mx-auto flex w-full max-w-3xl items-end gap-2"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            rows={1}
            placeholder={
              hasDocuments
                ? "Ask a question…"
                : "Upload a document first, then ask a question…"
            }
            className="max-h-40 flex-1 resize-none rounded-xl border border-ink-200 bg-white px-4 py-3 text-sm outline-none transition focus:border-anchor-400 dark:border-ink-800 dark:bg-ink-800/50"
          />
          {streaming ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="rounded-xl border border-ink-200 px-4 py-3 text-sm font-medium transition hover:bg-ink-100 dark:border-ink-800 dark:hover:bg-ink-800"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded-xl bg-anchor-500 px-5 py-3 text-sm font-medium text-white transition hover:bg-anchor-600 disabled:opacity-40"
            >
              Ask
            </button>
          )}
        </form>
      </div>
    </main>
  );
}
