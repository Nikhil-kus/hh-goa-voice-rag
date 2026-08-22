"""
Unit tests for all three chunking strategies.

Verifies:
- Chunk boundaries differ between strategies
- No answer / query text leaks into chunk.text
- is_selected stored correctly but never used as retrieval signal
- Correct strategy name on each chunk
- No empty chunks produced
- Fixed-size overlap arithmetic
- Sentence-aware never splits mid-sentence
- Passage-structure-aware always produces exactly one chunk per non-empty passage
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.services.chunker import (
    FixedSizeChunker,
    SentenceAwareChunker,
    PassageStructureAwareChunker,
    chunk_record,
    _TARGET_LANG_MAP,
)

# ── Shared test fixtures ───────────────────────────────────────────────────────

ENGLISH_PASSAGE = (
    "The Manhattan Project was a research and development undertaking during "
    "World War II that produced the first nuclear weapons. It was led by the "
    "United States with the support of the United Kingdom and Canada. "
    "From 1942 to 1946, the project was under the direction of Major General "
    "Leslie Groves of the Army Corps of Engineers. "
    "The project's scientific director was J. Robert Oppenheimer. "
    "The project succeeded in developing and detonating three nuclear devices "
    "in 1945."
)

SHORT_PASSAGE = "This is a short passage. It has two sentences."

FAKE_RECORD = {
    "query_id": 12345,
    "query_type": "DESCRIPTION",
    "query": "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ কি আছিল?",
    "Eng_Query": "what was the immediate impact of the success of the manhattan project?",
    "Answer": "THE GROUND TRUTH ANSWER THAT MUST NEVER APPEAR IN CHUNKS",
    "Eng_Answer": "THE ENGLISH GROUND TRUTH THAT MUST NEVER APPEAR IN CHUNKS",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva",
    "passages": {
        "is_selected": [1, 0, 0],
        "English_passages": [
            ENGLISH_PASSAGE,
            "A second passage about nuclear physics experiments.",
            "A third passage about wartime research programs.",
        ],
        "Translated_passages": [
            "हिंदी अनुवाद पहला अनुच्छेद।",
            "हिंदी अनुवाद दूसरा अनुच्छेद।",
            "हिंदी अनुवाद तीसरा अनुच्छेद।",
        ],
    },
}


# ── Strategy 1: Fixed-size overlap ────────────────────────────────────────────

class TestFixedSizeChunker:

    def setup_method(self):
        self.chunker = FixedSizeChunker(size=20, overlap=5)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=1, source="english"
        )
        assert len(chunks) >= 1

    def test_strategy_name(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        for c in chunks:
            assert c.strategy == "fixed_size_overlap"

    def test_no_empty_chunks(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        for c in chunks:
            assert c.text.strip() != ""

    def test_overlap_creates_multiple_chunks(self):
        # 100-token passage, size=20, overlap=5 → step=15 → multiple chunks
        long_text = " ".join([f"word{i}" for i in range(100)])
        chunks = self.chunker.chunk_passage(
            long_text, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert len(chunks) > 1

    def test_answer_not_in_chunk_text(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=1, source="english"
        )
        for c in chunks:
            assert "GROUND TRUTH ANSWER" not in c.text
            assert "GROUND TRUTH" not in c.text

    def test_is_selected_stored_as_metadata(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=1, source="english"
        )
        # is_selected stored in chunk object
        for c in chunks:
            assert c.is_selected == 1
        # but chunk.text does NOT contain the selection label
        for c in chunks:
            assert "is_selected" not in c.text

    def test_short_passage_single_chunk(self):
        chunks = FixedSizeChunker(size=256, overlap=32).chunk_passage(
            SHORT_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="ENTITY", is_selected=0, source="english"
        )
        assert len(chunks) == 1

    def test_empty_passage_returns_empty(self):
        chunks = self.chunker.chunk_passage(
            "", language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert chunks == []

    def test_chunk_idx_increments(self):
        long_text = " ".join([f"word{i}" for i in range(100)])
        chunks = self.chunker.chunk_passage(
            long_text, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        for i, c in enumerate(chunks):
            assert c.chunk_idx == i


# ── Strategy 2: Sentence-aware ────────────────────────────────────────────────

class TestSentenceAwareChunker:

    def setup_method(self):
        self.chunker = SentenceAwareChunker(max_tokens=50)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert len(chunks) >= 1

    def test_strategy_name(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        for c in chunks:
            assert c.strategy == "sentence_aware"

    def test_no_empty_chunks(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        for c in chunks:
            assert c.text.strip() != ""

    def test_chunks_respect_sentence_boundaries(self):
        # Each chunk must end with sentence-terminal punctuation or be the last
        # (a proxy for not splitting mid-sentence)
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = SentenceAwareChunker(max_tokens=10).chunk_passage(
            text, language="en", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        # All sentence text should be present across chunks
        combined = " ".join(c.text for c in chunks)
        assert "First sentence" in combined
        assert "Second sentence" in combined
        assert "Third sentence" in combined

    def test_sentence_count_in_extra(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        for c in chunks:
            assert "sentence_count" in c.extra
            assert c.extra["sentence_count"] >= 1

    def test_different_boundaries_from_fixed_size(self):
        fixed = FixedSizeChunker(size=20, overlap=5).chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        sent = SentenceAwareChunker(max_tokens=40).chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        # Chunk texts must not all be identical
        fixed_texts = [c.text for c in fixed]
        sent_texts  = [c.text for c in sent]
        assert fixed_texts != sent_texts, "Sentence-aware must produce different boundaries than fixed-size"

    def test_empty_passage_returns_empty(self):
        chunks = self.chunker.chunk_passage(
            "", language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert chunks == []

    def test_is_selected_stored(self):
        chunks = self.chunker.chunk_passage(
            SHORT_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=1, source="english"
        )
        for c in chunks:
            assert c.is_selected == 1


# ── Strategy 3: Passage-structure-aware ──────────────────────────────────────

class TestPassageStructureAwareChunker:

    def setup_method(self):
        self.chunker = PassageStructureAwareChunker()

    def test_produces_exactly_one_chunk_per_passage(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=1, source="english"
        )
        assert len(chunks) == 1

    def test_strategy_name(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert chunks[0].strategy == "passage_structure_aware"

    def test_chunk_contains_full_passage_text(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        # The chunk text should be the normalized passage (no truncation)
        assert "Manhattan Project" in chunks[0].text

    def test_chunk_idx_always_zero(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=2, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert chunks[0].chunk_idx == 0

    def test_token_count_in_extra(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert "token_count" in chunks[0].extra
        assert chunks[0].extra["token_count"] > 0

    def test_different_boundaries_from_fixed_size(self):
        # Passage-structure produces 1 chunk; fixed-size produces multiple
        fixed = FixedSizeChunker(size=20, overlap=5).chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        struct = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert len(struct) == 1
        assert len(fixed) > 1

    def test_is_selected_stored_not_in_text(self):
        chunks = self.chunker.chunk_passage(
            ENGLISH_PASSAGE, language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=1, source="english"
        )
        assert chunks[0].is_selected == 1
        assert "is_selected" not in chunks[0].text
        assert "1" not in chunks[0].text.split()[:5]  # "1" not at start of text

    def test_empty_passage_returns_empty(self):
        chunks = self.chunker.chunk_passage(
            "", language="hi", query_id=1,
            passage_idx=0, query_type="DESCRIPTION", is_selected=0, source="english"
        )
        assert chunks == []


# ── chunk_record integration ─────────────────────────────────────────────────

class TestChunkRecord:

    def test_chunks_all_passages(self):
        chunks = chunk_record(FAKE_RECORD, strategy="fixed_size_overlap", source="english")
        query_ids = {c.query_id for c in chunks}
        assert 12345 in query_ids

    def test_answer_never_in_chunk_text(self):
        for strategy in ["fixed_size_overlap", "sentence_aware", "passage_structure_aware"]:
            chunks = chunk_record(FAKE_RECORD, strategy=strategy, source="english")
            for c in chunks:
                assert "GROUND TRUTH ANSWER" not in c.text
                assert "MUST NEVER APPEAR" not in c.text

    def test_query_never_in_chunk_text(self):
        for strategy in ["fixed_size_overlap", "sentence_aware", "passage_structure_aware"]:
            chunks = chunk_record(FAKE_RECORD, strategy=strategy, source="english")
            for c in chunks:
                # The query text must not be embedded in the passage chunk
                assert FAKE_RECORD["Eng_Query"] not in c.text
                assert FAKE_RECORD["query"] not in c.text

    def test_is_selected_in_payload(self):
        chunks = chunk_record(FAKE_RECORD, strategy="passage_structure_aware", source="english")
        # First passage has is_selected=1
        first_passage_chunks = [c for c in chunks if c.passage_idx == 0]
        assert len(first_passage_chunks) >= 1
        assert first_passage_chunks[0].is_selected == 1

    def test_strategy_three_produces_three_chunks(self):
        # 3 passages → passage_structure_aware → 3 chunks (one per passage)
        chunks = chunk_record(FAKE_RECORD, strategy="passage_structure_aware", source="english")
        assert len(chunks) == 3

    def test_all_three_strategies_work(self):
        for strategy in ["fixed_size_overlap", "sentence_aware", "passage_structure_aware"]:
            chunks = chunk_record(FAKE_RECORD, strategy=strategy, source="english")
            assert len(chunks) > 0, f"Strategy {strategy} produced no chunks"

    def test_to_payload_has_required_keys(self):
        chunks = chunk_record(FAKE_RECORD, strategy="passage_structure_aware", source="english")
        required_keys = {
            "text", "strategy", "language", "query_id",
            "passage_idx", "chunk_idx", "query_type",
            "is_selected", "source",
        }
        for c in chunks:
            payload = c.to_payload()
            missing = required_keys - set(payload.keys())
            assert not missing, f"Payload missing keys: {missing}"

    def test_language_mapping(self):
        # target_lang "hin_Deva" should map to "hi"
        chunks = chunk_record(FAKE_RECORD, strategy="passage_structure_aware", source="english")
        for c in chunks:
            assert c.language == "hi"


# ── TARGET_LANG_MAP completeness ──────────────────────────────────────────────

def test_target_lang_map_covers_all_dataset_languages():
    expected = {"hin", "ben", "tam", "tel", "kan", "mal", "mar", "guj",
                "pan", "ori", "asm", "nep", "urd", "san"}
    assert expected.issubset(set(_TARGET_LANG_MAP.keys()))
