"use client";
import { useCallback, useState } from "react";
import AnswerPanel from "@/components/AnswerPanel";
import ChunkingSelector from "@/components/ChunkingSelector";
import LatencyPanel from "@/components/LatencyPanel";
import SourceCards from "@/components/SourceCards";
import TranscriptDisplay from "@/components/TranscriptDisplay";
import VoiceRecorder from "@/components/VoiceRecorder";
import { queryRAG, transcribeAudio } from "@/lib/api";
import type {
  AppPhase,
  ChunkingStrategy,
  LatencyBreakdown,
  QueryResponse,
  RetrievedSource,
} from "@/lib/types";

export default function Home() {
  const [phase, setPhase] = useState<AppPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<ChunkingStrategy>("passage_structure_aware");

  // Transcription state
  const [transcript, setTranscript] = useState("");
  const [languageCode, setLanguageCode] = useState<string | undefined>(undefined);
  const [sttLatencyMs, setSttLatencyMs] = useState<number | undefined>(undefined);

  // Query result state
  const [answer, setAnswer] = useState<string | undefined>(undefined);
  const [refused, setRefused] = useState(false);
  const [refusalReason, setRefusalReason] = useState<QueryResponse["refusal_reason"]>(undefined);
  const [refusalMessage, setRefusalMessage] = useState<string | undefined>(undefined);
  const [sources, setSources] = useState<RetrievedSource[]>([]);
  const [latency, setLatency] = useState<LatencyBreakdown | undefined>(undefined);

  const reset = () => {
    setPhase("idle");
    setError(null);
    setTranscript("");
    setLanguageCode(undefined);
    setSttLatencyMs(undefined);
    setAnswer(undefined);
    setRefused(false);
    setRefusalReason(undefined);
    setRefusalMessage(undefined);
    setSources([]);
    setLatency(undefined);
  };

  const handleError = useCallback((msg: string) => {
    setError(msg);
    setPhase("error");
  }, []);

  const handleRecordingComplete = useCallback(
    async (blob: Blob) => {
      setPhase("transcribing");
      setError(null);

      // Step 1: STT
      let txResult;
      try {
        txResult = await transcribeAudio(blob, "unknown");
      } catch (e) {
        handleError(e instanceof Error ? e.message : "Transcription failed");
        return;
      }

      const tx = txResult.transcript;
      const lang = txResult.language_code;
      const sttMs = txResult.latency_ms?.stt ?? txResult.latency_ms?.total ?? undefined;

      setTranscript(tx);
      setLanguageCode(lang);
      setSttLatencyMs(sttMs);

      // Step 2: RAG query
      setPhase("querying");
      let qResult: QueryResponse;
      try {
        qResult = await queryRAG(tx, lang, strategy);
      } catch (e) {
        handleError(e instanceof Error ? e.message : "Query failed");
        return;
      }

      setAnswer(qResult.answer);
      setRefused(qResult.refused);
      setRefusalReason(qResult.refusal_reason);
      setRefusalMessage(qResult.refusal_message);
      setSources(qResult.sources);
      setLatency(qResult.latency);
      setPhase("done");
    },
    [strategy, handleError]
  );

  const hasResult = phase === "done" || phase === "error";

  return (
    <main className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-3">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-slate-900">Voice RAG</h1>
            <p className="text-xs text-slate-400">HH Goa 2026 · MSMARCO-XI Indic</p>
          </div>
          <a
            href="https://huggingface.co/datasets/ai4bharat/MSMARCO-XI"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-brand-600 hover:underline"
          >
            Dataset ↗
          </a>
        </div>
      </header>

      <div className="max-w-2xl mx-auto px-4 py-8 space-y-6">

        {/* Chunking strategy selector */}
        <ChunkingSelector
          value={strategy}
          onChange={setStrategy}
          disabled={phase === "transcribing" || phase === "querying"}
        />

        {/* Voice recorder */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 flex flex-col items-center gap-4">
          <VoiceRecorder
            phase={phase}
            onRecordingComplete={handleRecordingComplete}
            onError={handleError}
          />
          {hasResult && (
            <button
              onClick={reset}
              className="text-sm text-slate-500 hover:text-slate-700 underline"
            >
              Ask another question
            </button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Transcript */}
        {transcript && (
          <TranscriptDisplay
            transcript={transcript}
            languageCode={languageCode}
            sttLatencyMs={sttLatencyMs}
          />
        )}

        {/* Answer or refusal */}
        {(answer != null || refused) && (
          <AnswerPanel
            answer={answer}
            refused={refused}
            refusalReason={refusalReason}
            refusalMessage={refusalMessage}
          />
        )}

        {/* Sources */}
        {sources.length > 0 && <SourceCards sources={sources} />}

        {/* Latency */}
        {(latency || phase === "done") && (
          <LatencyPanel latency={latency} sttLatencyMs={sttLatencyMs} />
        )}

        {/* Dataset disclaimer */}
        <p className="text-xs text-slate-400 text-center leading-relaxed">
          Index: ~4×1,000 records per language (hi, bn, ta, kn) from MSMARCO-XI
          validation splits · English passages indexed · Limitation: 1.2% of full dataset
        </p>
      </div>
    </main>
  );
}
