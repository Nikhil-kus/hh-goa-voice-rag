import type {
  AllBenchmarkResults,
  ChunkingStrategy,
  QueryResponse,
  TranscribeResponse,
} from "./types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Transcribe ────────────────────────────────────────────────────────────────

export async function transcribeAudio(
  audioBlob: Blob,
  languageCode = "unknown"
): Promise<TranscribeResponse> {
  const form = new FormData();
  form.append("file", audioBlob, "recording.webm");
  form.append("language_code", languageCode);

  const res = await fetch(`${API_URL}/api/transcribe`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Transcription failed (${res.status})`);
  }
  return res.json();
}

// ── Query ─────────────────────────────────────────────────────────────────────

export async function queryRAG(
  transcript: string,
  languageCode: string | undefined,
  strategy: ChunkingStrategy
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      transcript,
      language_code: languageCode ?? null,
      strategy,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Query failed (${res.status})`);
  }
  return res.json();
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

export async function getBenchmarkResults(): Promise<AllBenchmarkResults> {
  const res = await fetch(`${API_URL}/api/benchmark/all`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark results (${res.status})`);
  return res.json();
}

export async function triggerLatencyBenchmark(): Promise<void> {
  await fetch(`${API_URL}/api/benchmark/latency`, { method: "POST" });
}

export async function triggerRetrievalBenchmark(): Promise<void> {
  await fetch(`${API_URL}/api/benchmark/retrieval`, { method: "POST" });
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Backend unavailable");
  return res.json();
}
