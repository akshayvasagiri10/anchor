"use client";

import { Fragment } from "react";

const CITATION = /\[(\d+)\]/g;

/**
 * Renders an answer, turning inline `[n]` markers into clickable chips.
 *
 * The chip is the whole point of the project: a citation the user cannot
 * verify is indistinguishable from a hallucination, so every marker resolves
 * to a source card they can actually read.
 */
export function AnswerText({
  text,
  maxCitation,
  onCite,
}: {
  text: string;
  maxCitation: number;
  onCite: (id: number) => void;
}) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  for (const match of text.matchAll(CITATION)) {
    const index = match.index ?? 0;
    const n = Number(match[1]);

    if (index > cursor) {
      parts.push(<Fragment key={key++}>{text.slice(cursor, index)}</Fragment>);
    }

    // A number outside the source range means the model invented a citation.
    // Render it as plain text rather than a dead link — silently dropping it
    // would hide a real failure from whoever is evaluating the output.
    if (n >= 1 && n <= maxCitation) {
      parts.push(
        <button
          key={key++}
          type="button"
          onClick={() => onCite(n)}
          title={`Jump to source ${n}`}
          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-anchor-500/15 px-1.5 align-baseline text-[11px] font-semibold text-anchor-600 transition hover:bg-anchor-500/30 dark:text-anchor-400"
        >
          {n}
        </button>,
      );
    } else {
      parts.push(<Fragment key={key++}>{match[0]}</Fragment>);
    }
    cursor = index + match[0].length;
  }

  if (cursor < text.length) {
    parts.push(<Fragment key={key++}>{text.slice(cursor)}</Fragment>);
  }

  return <>{parts}</>;
}
