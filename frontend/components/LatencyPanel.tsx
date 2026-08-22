"use client";
import { useEffect, useState } from "react";
import type {
  AllBenchmarkResults,
  LatencyBreakdown,
  LatencyBenchmarkResult,
  RetrievalBenchmarkResult,
} from "@/lib/types";
import { getBenchmarkResults } from "@/lib/api";

interface Props {
  latency?: LatencyBreakdown;
  sttLatencyMs?: number;
}

const STAGE_LABELS: Record<string, string> = {
  query_normalization_ms: "Query norm",
  guardrail_pre_ms:       "Guardrails",
  embedding_ms:           "Embedding",
  vector_retrieval_ms:    "Vector search",
  reranking_ms:           "Reranking",
  evidence_scoring_ms:    "Evidence gate",
  generation_ms:          "LLM generation",
  grounding_check_ms:     "Grounding check",
  total_ms:               "Total",
};

function Bar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-slate-500 w-16 text-right">
        {value.toFixed(1)}ms
      </span>
    </div>
  );
}

export default function LatencyPanel({ latency, sttLatencyMs }: Props) {
  const [benchmarks, setBenchmarks] = useState<AllBenchmarkResults | null>(null);
  const [showBench, setShowBench] = useState(false);
  const [loading, setLoading] = useState(false);

  const loadBenchmarks = async () => {
    setLoading(true);
    try {
      const data = await getBenchmarkResults();
      setBenchmarks(data);
    } catch {
      setBenchmarks(null);
    } finally {
      setLoading(false);
    }
  };

  if (!latency && !benchmarks) return null;

  const stageEntries = latency
    ? Object.entries(STAGE_LABELS)
        .map(([key, label]) => {
          const val = latency[key as keyof LatencyBreakdown];
          return val != null ? { key, label, val } : null;
        })
        .filter(Boolean) as { key: string; label: string; val: number }[]
    : [];

  const maxVal = stageEntries.reduce((m, s) => Math.max(m, s.val), 1);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
          Latency
        </p>
        <button
          onClick={() => { setShowBench((v) => !v); if (!benchmarks && !showBench) loadBenchmarks(); }}
          className="text-xs text-brand-600 hover:underline"
        >
          {showBench ? "Hide benchmarks" : "Show P50/P70/P100"}
        </button>
      </div>

      {/* Per-request latency */}
      {latency && (
        <div className="space-y-2">
          {sttLatencyMs != null && (
            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-0.5">
                <span>STT (network)</span>
              </div>
              <Bar value={sttLatencyMs} max={Math.max(maxVal, sttLatencyMs)} />
            </div>
          )}
          {stageEntries.map(({ key, label, val }) => (
            <div key={key}>
              <div className="text-xs text-slate-500 mb-0.5">{label}</div>
              <Bar value={val} max={maxVal} />
            </div>
          ))}
        </div>
      )}

      {/* Benchmark P-values */}
      {showBench && (
        <div className="border-t border-slate-100 pt-4">
          {loading && <p className="text-xs text-slate-400">Loading…</p>}
          {!loading && !benchmarks && (
            <p className="text-xs text-slate-400">No benchmark data yet.</p>
          )}
          {!loading && benchmarks?.latency && (
            <BenchmarkTable latency={benchmarks.latency} retrieval={benchmarks.retrieval} />
          )}
        </div>
      )}
    </div>
  );
}

function BenchmarkTable({
  latency,
  retrieval,
}: {
  latency: LatencyBenchmarkResult[];
  retrieval: RetrievalBenchmarkResult[] | null;
}) {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-xs font-semibold text-slate-500 mb-2">Latency P50/P70/P100 (ms)</p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="pb-1 pr-3">Strategy</th>
                <th className="pb-1 pr-3">Stage</th>
                <th className="pb-1 pr-3 text-right">P50</th>
                <th className="pb-1 pr-3 text-right">P70</th>
                <th className="pb-1 text-right">P100</th>
              </tr>
            </thead>
            <tbody>
              {latency.flatMap((r) =>
                Object.entries(r.stages)
                  .filter(([k]) => k === "total_rag" || k === "embedding" || k === "vector_retrieval")
                  .map(([stage, s]) => (
                    <tr key={`${r.strategy}-${stage}`} className="border-t border-slate-50">
                      <td className="py-1 pr-3 text-slate-500">{r.strategy.replace(/_/g, " ")}</td>
                      <td className="py-1 pr-3">{stage.replace(/_/g, " ")}</td>
                      <td className="py-1 pr-3 text-right font-mono">{s.p50.toFixed(1)}</td>
                      <td className="py-1 pr-3 text-right font-mono">{s.p70.toFixed(1)}</td>
                      <td className="py-1 text-right font-mono">{s.p100.toFixed(1)}</td>
                    </tr>
                  ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {retrieval && (
        <div>
          <p className="text-xs font-semibold text-slate-500 mb-2">Retrieval Quality (Recall@K)</p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-400">
                  <th className="pb-1 pr-3">Strategy</th>
                  <th className="pb-1 pr-3 text-right">R@1</th>
                  <th className="pb-1 pr-3 text-right">R@3</th>
                  <th className="pb-1 pr-3 text-right">R@5</th>
                  <th className="pb-1 text-right">R@10</th>
                </tr>
              </thead>
              <tbody>
                {retrieval.map((r) => (
                  <tr key={r.strategy} className="border-t border-slate-50">
                    <td className="py-1 pr-3 text-slate-500">{r.strategy.replace(/_/g, " ")}</td>
                    <td className="py-1 pr-3 text-right font-mono">{(r.recall.recall_at_1 ?? 0).toFixed(3)}</td>
                    <td className="py-1 pr-3 text-right font-mono">{(r.recall.recall_at_3 ?? 0).toFixed(3)}</td>
                    <td className="py-1 pr-3 text-right font-mono">{(r.recall.recall_at_5 ?? 0).toFixed(3)}</td>
                    <td className="py-1 text-right font-mono">{(r.recall.recall_at_10 ?? 0).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
