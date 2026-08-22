"use client";
import type { ChunkingStrategy } from "@/lib/types";

const STRATEGIES: { value: ChunkingStrategy; label: string; description: string }[] = [
  {
    value: "passage_structure_aware",
    label: "Passage-Structure Aware",
    description: "Uses dataset passage boundaries + metadata",
  },
  {
    value: "sentence_aware",
    label: "Sentence-Aware",
    description: "Groups complete sentences up to token budget",
  },
  {
    value: "fixed_size_overlap",
    label: "Fixed-Size + Overlap",
    description: "256-token window, 32-token overlap",
  },
];

interface Props {
  value: ChunkingStrategy;
  onChange: (s: ChunkingStrategy) => void;
  disabled?: boolean;
}

export default function ChunkingSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Chunking Strategy
      </label>
      <div className="flex flex-col sm:flex-row gap-2">
        {STRATEGIES.map((s) => (
          <button
            key={s.value}
            onClick={() => onChange(s.value)}
            disabled={disabled}
            className={`flex-1 text-left px-3 py-2 rounded-lg border text-sm transition-colors
              ${value === s.value
                ? "border-brand-500 bg-brand-50 text-brand-700 font-medium"
                : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
              }
              ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          >
            <div className="font-medium">{s.label}</div>
            <div className="text-xs text-slate-400 mt-0.5">{s.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
