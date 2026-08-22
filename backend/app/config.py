"""
Central configuration — all values come from environment variables.
No hardcoded API keys anywhere in the codebase.
"""
from __future__ import annotations

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Sarvam STT ────────────────────────────────────────────────────────────
    sarvam_api_key: str = ""
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"
    stt_timeout_ms: int = 10_000

    # ── LLM (modular) ─────────────────────────────────────────────────────────
    llm_provider: str = "groq"          # groq | openai | together | gemini
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    openai_api_key: str = ""
    together_api_key: str = ""
    gemini_api_key: str = ""
    generation_timeout_ms: int = 15_000

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    # Local path used when Qdrant server is unavailable (embedded mode)
    qdrant_local_path: str = "./qdrant_data"
    use_qdrant_local: bool = True       # set False when Docker Qdrant is running

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    embedding_timeout_ms: int = 1_000

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 10
    evidence_threshold: float = 0.25   # calibrated for 40K-passage index; raise to 0.35 with full dataset
    grounding_threshold: float = 0.10  # ROUGE-1 recall; 0.10 calibrated for short factual answers
    reranker_enabled: bool = False
    retrieval_timeout_ms: int = 2_000

    # ── Dataset / Ingestion ───────────────────────────────────────────────────
    languages_to_index: str = "hi,bn,ta,te,kn"
    records_per_language: int = 5_000  # start small; increase after system works
    chunking_strategies: str = "fixed_size_overlap,sentence_aware,passage_structure_aware"

    # ── Guardrails ────────────────────────────────────────────────────────────
    supported_languages: str = "hi,bn,ta,te,kn"
    off_topic_threshold: float = 0.15  # min similarity to corpus centroid
    max_transcript_chars: int = 500

    # ── Timeouts ──────────────────────────────────────────────────────────────
    query_normalization_timeout_ms: int = 100
    guardrail_timeout_ms: int = 500
    evidence_scoring_timeout_ms: int = 100
    grounding_check_timeout_ms: int = 500

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # ── Computed helpers ──────────────────────────────────────────────────────
    @property
    def language_list(self) -> List[str]:
        return [l.strip() for l in self.languages_to_index.split(",") if l.strip()]

    @property
    def strategy_list(self) -> List[str]:
        return [s.strip() for s in self.chunking_strategies.split(",") if s.strip()]

    @property
    def supported_language_list(self) -> List[str]:
        return [l.strip() for l in self.supported_languages.split(",") if l.strip()]

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def collection_name(self, strategy: str) -> str:
        return f"msmarco_xi_{strategy}"


settings = Settings()
