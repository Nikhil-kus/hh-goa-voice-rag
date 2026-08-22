"""
Unit tests for the retriever service.
Qdrant search is mocked — no live Qdrant needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import ChunkingStrategy
from app.services.retriever import (
    language_code_to_iso,
    normalize_query,
    retrieve,
)
from app.utils.timing import StageTimer


# ── normalize_query ───────────────────────────────────────────────────────────

class TestNormalizeQuery:
    def test_strips_whitespace(self):
        assert normalize_query("  hello world  ") == "hello world"

    def test_collapses_spaces(self):
        assert normalize_query("hello   world") == "hello world"

    def test_unicode_nfc(self):
        # NFC normalization: composed form
        result = normalize_query("caf\u00e9")
        assert result == "caf\u00e9"

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_newlines_collapsed(self):
        assert normalize_query("hello\nworld\ttab") == "hello world tab"


# ── language_code_to_iso ──────────────────────────────────────────────────────

class TestLanguageCodeToIso:
    def test_sarvam_hi_in(self):
        assert language_code_to_iso("hi-IN") == "hi"

    def test_sarvam_bn_in(self):
        assert language_code_to_iso("bn-IN") == "bn"

    def test_sarvam_ta_in(self):
        assert language_code_to_iso("ta-IN") == "ta"

    def test_sarvam_kn_in(self):
        assert language_code_to_iso("kn-IN") == "kn"

    def test_none_returns_none(self):
        assert language_code_to_iso(None) is None

    def test_unknown_short_code(self):
        # "xx-XX" → strips to "xx" (2 chars)
        result = language_code_to_iso("xx-XX")
        assert result == "xx"

    def test_en_in(self):
        assert language_code_to_iso("en-IN") == "en"

    def test_bare_code(self):
        # bare "hi" without region → falls through to split → "hi"
        result = language_code_to_iso("hi")
        assert result == "hi"


# ── retrieve (mocked Qdrant) ──────────────────────────────────────────────────

def _make_mock_result(query_id: int, passage_idx: int, score: float,
                      is_selected: int = 0, lang: str = "hi") -> dict:
    return {
        "score": score,
        "payload": {
            "text": f"passage text for query {query_id} passage {passage_idx}",
            "language": lang,
            "query_id": query_id,
            "passage_idx": passage_idx,
            "chunk_idx": 0,
            "strategy": "passage_structure_aware",
            "query_type": "DESCRIPTION",
            "is_selected": is_selected,
            "source": "english",
        },
    }


MOCK_RESULTS = [
    _make_mock_result(101, 0, 0.82, is_selected=1),
    _make_mock_result(101, 1, 0.71, is_selected=0),
    _make_mock_result(202, 0, 0.65, is_selected=0),
    _make_mock_result(303, 2, 0.55, is_selected=1),
    _make_mock_result(404, 0, 0.40, is_selected=0),
]


@patch("app.services.retriever.search", return_value=MOCK_RESULTS)
@patch("app.services.retriever.embed_query", return_value=np.ones(384, dtype=np.float32) / np.sqrt(384))
class TestRetrieve:

    def test_returns_sources_and_max_score(self, mock_embed, mock_search):
        timer = StageTimer()
        sources, max_score = retrieve(
            query="what is the manhattan project",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="hi-IN",
            top_k=10,
            reranker_enabled=False,
            timer=timer,
        )
        assert len(sources) == len(MOCK_RESULTS)
        assert abs(max_score - 0.82) < 0.01

    def test_sources_have_no_is_selected_field(self, mock_embed, mock_search):
        """is_selected must NOT appear in RetrievedSource returned to API."""
        timer = StageTimer()
        sources, _ = retrieve(
            query="test query",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="hi-IN",
            top_k=10,
            reranker_enabled=False,
            timer=timer,
        )
        for s in sources:
            # RetrievedSource model has no is_selected field
            assert not hasattr(s, "is_selected") or s.__dict__.get("is_selected") is None

    def test_scores_are_rounded(self, mock_embed, mock_search):
        timer = StageTimer()
        sources, _ = retrieve(
            query="test",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="hi-IN",
            top_k=10,
            reranker_enabled=False,
            timer=timer,
        )
        for s in sources:
            # score should be a float with ≤ 4 decimal places
            assert isinstance(s.score, float)
            assert s.score == round(s.score, 4)

    def test_timer_records_stages(self, mock_embed, mock_search):
        timer = StageTimer()
        retrieve(
            query="test",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="hi-IN",
            top_k=10,
            reranker_enabled=False,
            timer=timer,
        )
        ms = timer.all_ms()
        assert "query_normalization" in ms
        assert "embedding" in ms
        assert "vector_retrieval" in ms
        assert ms["embedding"] >= 0
        assert ms["vector_retrieval"] >= 0

    def test_language_filter_applied(self, mock_embed, mock_search):
        """Verify search is called with the correct language filter."""
        timer = StageTimer()
        retrieve(
            query="test",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="hi-IN",
            top_k=5,
            reranker_enabled=False,
            timer=timer,
        )
        call_kwargs = mock_search.call_args
        assert call_kwargs[1].get("language_filter") == "hi" or \
               call_kwargs[0][3] == "hi"  # positional or keyword

    def test_unsupported_language_no_filter(self, mock_embed, mock_search):
        """Language not in supported set → no filter applied."""
        timer = StageTimer()
        retrieve(
            query="test",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="xx-XX",   # unsupported
            top_k=5,
            reranker_enabled=False,
            timer=timer,
        )
        call_kwargs = mock_search.call_args
        # language_filter should be None for unsupported languages
        lang_filter = (call_kwargs[1].get("language_filter") or
                       (call_kwargs[0][3] if len(call_kwargs[0]) > 3 else None))
        assert lang_filter is None

    def test_empty_results_returns_zero_score(self, mock_embed, mock_search):
        mock_search.return_value = []
        timer = StageTimer()
        sources, max_score = retrieve(
            query="test",
            strategy=ChunkingStrategy.passage_structure_aware,
            language_code="hi-IN",
            top_k=10,
            reranker_enabled=False,
            timer=timer,
        )
        assert sources == []
        assert max_score == 0.0
        mock_search.return_value = MOCK_RESULTS  # restore

    def test_reranker_enabled_still_returns_results(self, mock_embed, mock_search):
        timer = StageTimer()
        sources, max_score = retrieve(
            query="test",
            strategy=ChunkingStrategy.fixed_size_overlap,
            language_code="hi-IN",
            top_k=10,
            reranker_enabled=True,
            timer=timer,
        )
        assert len(sources) > 0
        assert max_score > 0


# ── StageTimer integration ────────────────────────────────────────────────────

class TestStageTimer:
    def test_records_multiple_stages(self):
        import time
        timer = StageTimer()
        with timer.stage("step_a"):
            time.sleep(0.005)
        with timer.stage("step_b"):
            time.sleep(0.005)
        ms = timer.all_ms()
        assert ms["step_a"] >= 4.0
        assert ms["step_b"] >= 4.0
        assert ms["total"] >= ms["step_a"] + ms["step_b"]

    def test_total_always_present(self):
        timer = StageTimer()
        ms = timer.all_ms()
        assert "total" in ms

    def test_manual_record(self):
        timer = StageTimer()
        timer.record("stt", 312.5)
        ms = timer.all_ms()
        assert ms["stt"] == 312.5
