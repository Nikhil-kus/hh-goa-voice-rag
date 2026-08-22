"use client";
import { useState } from "react";
import type { RetrievedSource } from "@/lib/types";

const LANG_FLAGS: Record<string, string> = {
  hi: "🇮🇳 Hindi", bn: "🇧🇩 Bengali", ta: "🇮🇳 Tamil",
  te: "🇮🇳 Telugu", kn: "🇮🇳 Kannada", en: "🇬🇧 English",
};

interface Props {
  sources: RetrievedSource[];
}

export default function SourceCards({ sources }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);
  if (!sources.length) return null;

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
        Retrieved Sources ({sources.length})
      </p>
      {sources.map((s, i) => (
        <div
          key={i}
          className="rounded-lg border border-slate-200 bg-white overflow-hidden"
        >
          <button
            onClick={() => setExpanded(expanded === i ? null : i)}
            className="w-full text-left px-4 py-3 flex items-center justify-between gap-3 hover:bg-slate-50 transition-colors"
          >
            <div className="flex items-center gap-3 min-w-0">
              {/* Score badge */}
              <span
                className={`shrink-0 text-xs font-mono font-semibold px-2 py-0.5 rounded
                  ${s.score >= 0.7
                    ? "bg-green-100 text-green-700"
                    : s.score >= 0.5
                    ? "bg-yellow-100 text-yellow-700"
                    : "bg-slate-100 text-slate-500"
                  }`}
              >
                {s.score.toFixed(3)}
              </span>
              {/* Language */}
              <span className="text-xs text-slate-400 shrink-0">
                {LANG_FLAGS[s.language] ?? s.language}
              </span>
              {/* Passage preview */}
              <span className="text-sm text-slate-600 truncate">
                {s.text.slice(0, 120)}
              </span>
            </div>
            <span className="text-slate-300 shrink-0 text-xs">
              {expanded === i ? "▲" : "▼"}
            </span>
          </button>

          {expanded === i && (
            <div className="px-4 pb-4 pt-1 border-t border-slate-100">
              <p className="text-sm text-slate-700 leading-relaxed">{s.text}</p>
              <div className="mt-2 flex gap-3 text-xs text-slate-400">
                <span>query_id: {s.query_id}</span>
                <span>passage: {s.passage_idx}</span>
                {s.query_type && <span>type: {s.query_type}</span>}
                <span>strategy: {s.strategy}</span>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
