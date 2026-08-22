"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { AppPhase } from "@/lib/types";

interface Props {
  phase: AppPhase;
  onRecordingComplete: (blob: Blob) => void;
  onError: (msg: string) => void;
}

export default function VoiceRecorder({ phase, onRecordingComplete, onError }: Props) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/ogg";

      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: mimeType });
        if (blob.size < 1000) {
          onError("Recording too short — please hold the button and speak.");
          return;
        }
        onRecordingComplete(blob);
      };

      recorder.start(100); // collect data every 100ms
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("Permission") || msg.includes("denied")) {
        onError("Microphone permission denied. Please allow microphone access.");
      } else {
        onError(`Could not start recording: ${msg}`);
      }
    }
  }, [onRecordingComplete, onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  // Auto-stop after 28s (Sarvam limit is 30s)
  useEffect(() => {
    if (seconds >= 28 && isRecording) stopRecording();
  }, [seconds, isRecording, stopRecording]);

  const isDisabled = phase !== "idle" && phase !== "done" && phase !== "error";
  const busy = phase === "transcribing" || phase === "querying";

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Record button */}
      <button
        onMouseDown={startRecording}
        onMouseUp={stopRecording}
        onTouchStart={(e) => { e.preventDefault(); startRecording(); }}
        onTouchEnd={(e) => { e.preventDefault(); stopRecording(); }}
        disabled={isDisabled || busy}
        aria-label={isRecording ? "Stop recording" : "Hold to record"}
        className={`
          w-20 h-20 rounded-full flex items-center justify-center
          transition-all duration-150 select-none
          focus:outline-none focus:ring-4 focus:ring-brand-500/40
          ${isRecording
            ? "bg-red-500 scale-110 shadow-lg shadow-red-500/40"
            : isDisabled || busy
            ? "bg-slate-200 cursor-not-allowed"
            : "bg-brand-500 hover:bg-brand-600 shadow-md hover:shadow-lg cursor-pointer"
          }
        `}
      >
        {busy ? (
          <span className="w-7 h-7 border-2 border-white border-t-transparent rounded-full animate-spin" />
        ) : isRecording ? (
          <span className="w-5 h-5 bg-white rounded-sm" />
        ) : (
          <MicIcon />
        )}
      </button>

      {/* Status text */}
      <p className="text-sm text-slate-500 text-center">
        {busy
          ? phase === "transcribing"
            ? "Transcribing…"
            : "Generating answer…"
          : isRecording
          ? `Recording… ${seconds}s (release to send)`
          : isDisabled
          ? ""
          : "Hold to record"}
      </p>

      {/* Recording timer bar */}
      {isRecording && (
        <div className="w-48 h-1.5 bg-slate-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-red-500 transition-all duration-1000"
            style={{ width: `${Math.min((seconds / 28) * 100, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}

function MicIcon() {
  return (
    <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" />
    </svg>
  );
}
