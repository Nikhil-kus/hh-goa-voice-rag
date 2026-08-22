// ── Enums ─────────────────────────────────────────────────────────────────────

export type ChunkingStrategy =
  | "fixed_size_overlap"
  | "sentence_aware"
  | "passage_structure_aware";

export type RefusalReason =
  | "insufficient_evidence"
  | "grounding_failed"
  | "off_topic"
  | "unsafe_content"
  | "empty_transcript"
  | "stt_failed"
  | "unsupported_language"
  | "timeout"
  | "internal_error";

// ── API response types ────────────────────────────────────────────────────────

export interface TranscribeResponse {
  transcript: string;
  language_code: string;
  latency_ms: Record<string, number>;
  request_id?: string;
}

export interface RetrievedSource {
  text: string;
  language: string;
  query_id: number;
  passage_idx: number;
  score: number;
  strategy: string;
  query_type?: string;
}

export interface LatencyBreakdown {
  query_normalization_ms?: number;
  guardrail_pre_ms?: number;
  embedding_ms?: number;
  vector_retrieval_ms?: number;
  reranking_ms?: number;
  evidence_scoring_ms?: number;
  generation_ms?: number;
  grounding_check_ms?: number;
  total_ms: number;
}

export interface QueryResponse {
  answer?: string;
  refused: boolean;
  refusal_reason?: RefusalReason;
  refusal_message?: string;
  sources: RetrievedSource[];
  transcript: string;
  language_code?: string;
  strategy: string;
  latency: LatencyBreakdown;
}

// ── Benchmark types ───────────────────────────────────────────────────────────

export interface StagePct {
  p50: number;
  p70: number;
  p100: number;
  mean: number;
  n: number;
}

export interface LatencyBenchmarkResult {
  strategy: string;
  n_queries: number;
  n_errors: number;
  stages: Record<string, StagePct>;
}

export interface RetrievalBenchmarkResult {
  strategy: string;
  n_queries: number;
  n_evaluated: number;
  n_errors: number;
  languages_evaluated: Record<string, number>;
  recall: {
    recall_at_1?: number;
    recall_at_3?: number;
    recall_at_5?: number;
    recall_at_10?: number;
  };
}

export interface AllBenchmarkResults {
  latency: LatencyBenchmarkResult[] | null;
  retrieval: RetrievalBenchmarkResult[] | null;
}

// ── UI state ──────────────────────────────────────────────────────────────────

export type AppPhase =
  | "idle"
  | "recording"
  | "transcribing"
  | "querying"
  | "done"
  | "error";
