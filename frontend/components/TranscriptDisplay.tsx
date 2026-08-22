"use client";

const LANG_NAMES: Record<string, string> = {
  "hi-IN": "Hindi", "bn-IN": "Bengali", "ta-IN": "Tamil",
  "te-IN": "Telugu", "kn-IN": "Kannada", "ml-IN": "Malayalam",
  "mr-IN": "Marathi", "gu-IN": "Gujarati", "en-IN": "English (India)",
  "unknown": "Auto-detected",
};

interface Props {
  transcript: string;
  languageCode?: string;
  sttLatencyMs?: number;
}

export default function TranscriptDisplay({ transcript, languageCode, sttLatencyMs }: Props) {
  if (!transcript) return null;
  const langName = languageCode ? (LANG_NAMES[languageCode] ?? languageCode) : null;

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
          Transcript
        </span>
        <div className="flex items-center gap-2">
          {langName && (
            <span className="text-xs bg-brand-50 text-brand-700 border border-brand-200 rounded-full px-2 py-0.5">
              {langName}
            </span>
          )}
          {sttLatencyMs != null && (
            <span className="text-xs text-slate-400">STT {sttLatencyMs.toFixed(0)}ms</span>
          )}
        </div>
      </div>
      <p className="text-slate-800 text-base leading-relaxed">{transcript}</p>
    </div>
  );
}
