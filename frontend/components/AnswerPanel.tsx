"use client";
import type { RefusalReason } from "@/lib/types";

const REFUSAL_LABELS: Record<RefusalReason, string> = {
  insufficient_evidence: "Insufficient evidence",
  grounding_failed:      "Answer not grounded",
  off_topic:             "Off-topic query",
  unsafe_content:        "Unsafe content",
  empty_transcript:      "Empty transcript",
  stt_failed:            "STT failed",
  unsupported_language:  "Unsupported language",
  timeout:               "Timeout",
  internal_error:        "Internal error",
};

interface Props {
  answer?: string;
  refused: boolean;
  refusalReason?: RefusalReason;
  refusalMessage?: string;
}

export default function AnswerPanel({ answer, refused, refusalReason, refusalMessage }: Props) {
  if (!answer && !refused) return null;

  if (refused) {
    const label = refusalReason ? REFUSAL_LABELS[refusalReason] : "Refused";
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-amber-500 text-lg">⚠</span>
          <span className="text-xs font-semibold text-amber-600 uppercase tracking-wide">
            {label}
          </span>
        </div>
        <p className="text-amber-800 text-sm leading-relaxed">
          {refusalMessage ?? "I don't have sufficient information in the provided knowledge base to answer that."}
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-green-200 bg-green-50 p-4 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-green-500 text-lg">✓</span>
        <span className="text-xs font-semibold text-green-600 uppercase tracking-wide">
          Grounded Answer
        </span>
      </div>
      <p className="text-slate-800 text-base leading-relaxed whitespace-pre-wrap">{answer}</p>
    </div>
  );
}
