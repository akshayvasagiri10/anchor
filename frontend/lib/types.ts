export interface SourceCard {
  id: number;
  chunk_id: number;
  document_id: string;
  document_title: string;
  source: string;
  ordinal: number;
  text: string;
  score: number;
  matched_by: string[];
}

export interface DocumentSummary {
  id: string;
  title: string;
  source: string;
  n_chunks: number;
  n_chars: number;
  created_at: string;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  provider: string;
  model: string;
  local_generation: boolean;
  documents: number;
  chunks: number;
  embeddings: "ready" | "bm25-only";
  embedding_model: string | null;
  notes: string[];
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceCard[];
  stage?: "retrieving" | "thinking" | "writing" | "done";
  usage?: Usage;
  /** Citation numbers the model used that don't exist. Small local models
   *  do this; surfacing it is the point. */
  invalidCitations?: number[];
  model?: string;
  error?: string;
}

/** One event off the /api/chat SSE stream. */
export type StreamEvent =
  | { type: "sources"; sources: SourceCard[] }
  | { type: "status"; stage: "thinking" | "writing" }
  | { type: "token"; text: string }
  | {
      type: "done";
      model: string;
      stop_reason: string;
      usage: Usage;
      invalid_citations?: number[];
    }
  | { type: "error"; message: string };
