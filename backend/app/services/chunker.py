"""
Three genuinely different chunking strategies for MSMARCO-XI passages.

Key invariants (enforced throughout):
- NEVER include ground-truth Answer in chunk text
- NEVER include the original query text in the embedded chunk
- is_selected is stored as metadata ONLY — never used for retrieval ranking
- query_id is stored for offline evaluation linkage only

Strategy 1 — fixed_size_overlap
    Splits passage text into fixed token-count windows with overlap.
    Chunk boundaries are arbitrary (every N tokens).

Strategy 2 — sentence_aware
    Splits on sentence boundaries then groups sentences into chunks
    up to a token budget. Never splits mid-sentence.

Strategy 3 — passage_structure_aware
    Treats the entire dataset passage as one atomic semantic unit.
    Preserves full passage metadata. Chunk boundary = passage boundary.
    Genuinely different: boundaries come from the dataset's own structure,
    not from an arbitrary token window.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# nltk for sentence splitting — downloaded lazily
try:
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single chunk ready for embedding and upsert to Qdrant."""
    text: str                       # the text to embed — passage content only
    strategy: str
    language: str                   # e.g. "hi"
    query_id: int                   # for offline evaluation linkage
    passage_idx: int                # which passage within the record
    chunk_idx: int                  # which chunk within the passage (0-based)
    query_type: str                 # metadata only — not used as hard filter
    is_selected: int                # 0 or 1 — stored for evaluation, NOT retrieval
    source: str                     # "english" or "translated"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Qdrant point payload — all fields except the embedding vector."""
        return {
            "text": self.text,
            "strategy": self.strategy,
            "language": self.language,
            "query_id": self.query_id,
            "passage_idx": self.passage_idx,
            "chunk_idx": self.chunk_idx,
            "query_type": self.query_type,
            "is_selected": self.is_selected,   # stored, never boosted in retrieval
            "source": self.source,
            **self.extra,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Unicode NFC normalize and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _approx_token_count(text: str) -> int:
    """
    Approximate token count by whitespace splitting.
    Close enough for chunking decisions without loading a tokenizer.
    """
    return len(text.split())


def _split_into_tokens(text: str) -> List[str]:
    return text.split()


def _ensure_nltk_punkt() -> None:
    if not _NLTK_AVAILABLE:
        raise ImportError("nltk is required for sentence_aware chunking. pip install nltk")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _sentence_split(text: str, language: str) -> List[str]:
    """
    Split text into sentences.
    Uses nltk for all languages (indicnlp is optional enhancement).
    Falls back to period-splitting if nltk unavailable.
    """
    _ensure_nltk_punkt()
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Hard fallback: split on ". " or "। " (Devanagari danda)
        parts = re.split(r'(?<=[.।?!])\s+', text)
        return [p.strip() for p in parts if p.strip()]


# ── Strategy 1: Fixed-size with overlap ───────────────────────────────────────

class FixedSizeChunker:
    """
    Slide a window of `size` tokens over the passage with `overlap` tokens
    carried forward. Chunk boundaries are arbitrary token positions.
    """
    STRATEGY = "fixed_size_overlap"

    def __init__(self, size: int = 256, overlap: int = 32) -> None:
        assert overlap < size, "overlap must be less than size"
        self.size = size
        self.overlap = overlap

    def chunk_passage(
        self,
        passage_text: str,
        language: str,
        query_id: int,
        passage_idx: int,
        query_type: str,
        is_selected: int,
        source: str,
    ) -> List[Chunk]:
        text = _normalize_text(passage_text)
        if not text:
            return []

        tokens = _split_into_tokens(text)
        chunks: List[Chunk] = []
        step = self.size - self.overlap
        start = 0
        chunk_idx = 0

        while start < len(tokens):
            window = tokens[start : start + self.size]
            chunk_text = " ".join(window)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    strategy=self.STRATEGY,
                    language=language,
                    query_id=query_id,
                    passage_idx=passage_idx,
                    chunk_idx=chunk_idx,
                    query_type=query_type,
                    is_selected=is_selected,
                    source=source,
                    extra={"token_start": start, "token_end": start + len(window)},
                )
            )
            chunk_idx += 1
            start += step
            if start >= len(tokens):
                break

        return chunks


# ── Strategy 2: Sentence-aware ────────────────────────────────────────────────

class SentenceAwareChunker:
    """
    Groups complete sentences into chunks up to `max_tokens`.
    Never splits mid-sentence. Produces variable-length, linguistically
    coherent chunks — genuinely different boundaries from Strategy 1.
    """
    STRATEGY = "sentence_aware"

    def __init__(self, max_tokens: int = 200) -> None:
        self.max_tokens = max_tokens

    def chunk_passage(
        self,
        passage_text: str,
        language: str,
        query_id: int,
        passage_idx: int,
        query_type: str,
        is_selected: int,
        source: str,
    ) -> List[Chunk]:
        text = _normalize_text(passage_text)
        if not text:
            return []

        sentences = _sentence_split(text, language)
        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for sent in sentences:
            sent_tokens = _approx_token_count(sent)

            # If a single sentence exceeds budget, emit it alone
            if sent_tokens > self.max_tokens:
                if current_sentences:
                    chunks.append(self._make_chunk(
                        current_sentences, chunk_idx, language,
                        query_id, passage_idx, query_type, is_selected, source,
                    ))
                    chunk_idx += 1
                    current_sentences = []
                    current_tokens = 0
                chunks.append(self._make_chunk(
                    [sent], chunk_idx, language,
                    query_id, passage_idx, query_type, is_selected, source,
                ))
                chunk_idx += 1
                continue

            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                chunks.append(self._make_chunk(
                    current_sentences, chunk_idx, language,
                    query_id, passage_idx, query_type, is_selected, source,
                ))
                chunk_idx += 1
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sent)
            current_tokens += sent_tokens

        if current_sentences:
            chunks.append(self._make_chunk(
                current_sentences, chunk_idx, language,
                query_id, passage_idx, query_type, is_selected, source,
            ))

        return chunks

    def _make_chunk(
        self,
        sentences: List[str],
        chunk_idx: int,
        language: str,
        query_id: int,
        passage_idx: int,
        query_type: str,
        is_selected: int,
        source: str,
    ) -> Chunk:
        return Chunk(
            text=" ".join(sentences),
            strategy=self.STRATEGY,
            language=language,
            query_id=query_id,
            passage_idx=passage_idx,
            chunk_idx=chunk_idx,
            query_type=query_type,
            is_selected=is_selected,
            source=source,
            extra={"sentence_count": len(sentences)},
        )


# ── Strategy 3: Passage-structure-aware ───────────────────────────────────────

class PassageStructureAwareChunker:
    """
    Treats each dataset passage as one atomic semantic unit.

    Chunk boundaries = the dataset's own passage boundaries.
    This is genuinely different from Strategies 1 & 2:
    - No sub-passage splitting
    - Boundaries come from dataset structure, not token arithmetic
    - Full passage metadata (query_type, language) preserved
    - is_selected stored as metadata for evaluation ONLY — not used for ranking

    Label-leakage prevention:
    - Does NOT embed the ground-truth answer
    - Does NOT embed the original query text
    - query_id retained for offline evaluation linkage only
    """
    STRATEGY = "passage_structure_aware"

    def chunk_passage(
        self,
        passage_text: str,
        language: str,
        query_id: int,
        passage_idx: int,
        query_type: str,
        is_selected: int,
        source: str,
    ) -> List[Chunk]:
        text = _normalize_text(passage_text)
        if not text:
            return []

        return [
            Chunk(
                text=text,
                strategy=self.STRATEGY,
                language=language,
                query_id=query_id,
                passage_idx=passage_idx,
                chunk_idx=0,          # always 0 — one chunk per passage
                query_type=query_type,
                is_selected=is_selected,
                source=source,
                extra={"token_count": _approx_token_count(text)},
            )
        ]


# ── Unified interface ──────────────────────────────────────────────────────────

_CHUNKERS = {
    "fixed_size_overlap": FixedSizeChunker(),
    "sentence_aware": SentenceAwareChunker(),
    "passage_structure_aware": PassageStructureAwareChunker(),
}


def get_chunker(strategy: str) -> FixedSizeChunker | SentenceAwareChunker | PassageStructureAwareChunker:
    if strategy not in _CHUNKERS:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {list(_CHUNKERS)}")
    return _CHUNKERS[strategy]


def chunk_record(
    record: Dict[str, Any],
    strategy: str,
    source: str = "english",   # "english" | "translated"
) -> List[Chunk]:
    """
    Chunk all passages in a single MSMARCO-XI record.

    Args:
        record:   one row from the dataset
        strategy: one of fixed_size_overlap | sentence_aware | passage_structure_aware
        source:   whether to chunk English or Translated passages

    Returns:
        flat list of Chunk objects across all passages in the record

    Label-leakage invariants:
        - Answer / Eng_Answer are NEVER included in chunk.text
        - query / Eng_Query are NEVER included in chunk.text
        - is_selected stored as metadata, never used for retrieval ranking
    """
    chunker = get_chunker(strategy)
    passages = record.get("passages", {})
    english_passages: List[str] = passages.get("English_passages", [])
    translated_passages: List[str] = passages.get("Translated_passages", [])
    is_selected_list: List[int] = passages.get("is_selected", [])

    query_id = int(record.get("query_id", 0))
    query_type = str(record.get("query_type", "UNKNOWN"))
    language = str(record.get("target_lang", "")).split("_")[0].lower()
    # target_lang is e.g. "hin_Deva" → strip to "hin"; map to ISO 639-1
    language = _TARGET_LANG_MAP.get(language, language)

    all_chunks: List[Chunk] = []

    passage_texts = english_passages if source == "english" else translated_passages
    n = len(passage_texts)

    for idx, ptext in enumerate(passage_texts):
        if not ptext or not ptext.strip():
            continue
        is_sel = int(is_selected_list[idx]) if idx < len(is_selected_list) else 0
        chunks = chunker.chunk_passage(
            passage_text=ptext,
            language=language,
            query_id=query_id,
            passage_idx=idx,
            query_type=query_type,
            is_selected=is_sel,
            source=source,
        )
        all_chunks.extend(chunks)

    return all_chunks


# Map 3-letter ISO 639-3 codes from target_lang field to ISO 639-1
_TARGET_LANG_MAP: Dict[str, str] = {
    "hin": "hi",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "guj": "gu",
    "pan": "pa",
    "ori": "or",
    "asm": "as",
    "nep": "ne",
    "urd": "ur",
    "san": "sa",
}
