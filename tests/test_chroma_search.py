"""Tests for multi-term (AND) search in ChromaDbManager.search().

The embedding model and the translator are replaced by lookup tables, so the
vectors are known exactly: each word is one axis, and a product's cosine distance
to a term is easy to work out by hand. ChromaDB itself is real.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from managers.chroma_db_manager import (  # noqa: E402
    ChromaDbManager,
    join_search_terms,
    split_search_terms,
)

AXES = {"dress": 0, "red": 1, "forest": 2}


def _vector(text):
    v = np.zeros(len(AXES), dtype=np.float32)
    for word in text.split():
        v[AXES[word]] = 1.0
    return v / np.linalg.norm(v)


def fake_embeddings(texts, is_query=False):
    if isinstance(texts, str):
        return _vector(texts)
    return np.stack([_vector(t) for t in texts])


JA_EN = {"赤": "red", "ドレス": "dress", "森": "forest"}


def fake_translate(text, known_words=None):
    english = JA_EN.get(text)
    return (english, text) if english else (text, None)


@pytest.fixture
def db(tmp_path, monkeypatch):
    # Four products, so each term keeps its nearest two.
    monkeypatch.setenv("AND_TERM_FRACTION", "0.5")
    manager = ChromaDbManager(str(tmp_path / "chroma"), "products")
    products = {
        "red-dress": "red dress",
        "blue-dress": "dress",
        "red-car": "red",
        "woods": "forest",
    }
    manager.collection.add(
        ids=list(products),
        embeddings=[_vector(text).tolist() for text in products.values()],
        metadatas=[{"name": sku} for sku in products],
    )
    with patch("managers.chroma_db_manager.generate_embeddings", fake_embeddings), \
         patch("managers.chroma_db_manager.translate_query", fake_translate):
        yield manager


def _ranked(result):
    return [(r["id"], r["relevance_score"]) for r in result["results"]]


class TestSplitSearchTerms:
    @pytest.mark.parametrize("prompt, terms", [
        ("office", ["office"]),
        ("red dress", ["red", "dress"]),
        ("和室　背景", ["和室", "背景"]),           # full-width space
        ("  red   dress  ", ["red", "dress"]),
        ('"long hair" red', ["long hair", "red"]),
        ("“long hair” red", ["long hair", "red"]),  # curly quotes
        ("＂long hair＂ red", ["long hair", "red"]),  # full-width quotes
        ('"long hair red', ["long", "hair", "red"]),     # unmatched quote is dropped
        ('red red dress', ["red", "dress"]),
        ("", []),
        ('""', []),
    ])
    def test_split(self, prompt, terms):
        assert split_search_terms(prompt) == terms

    def test_join_reads_back_as_the_same_terms(self):
        terms = ["Japanese-style room", "background"]
        assert split_search_terms(join_search_terms(terms)) == terms


class TestAndSearch:
    def test_single_term_is_a_plain_nearest_neighbour_search(self, db):
        result = db.search("dress", limit=10, max_results=10)
        assert _ranked(result)[:2] == [("blue-dress", 1.0), ("red-dress", 0.7071)]
        assert result["query"] == "dress"
        assert result["terms"] == ["dress"]

    def test_every_term_must_match(self, db):
        # 'red' keeps red-car and red-dress, 'dress' keeps blue-dress and red-dress.
        result = db.search("red dress", limit=10, max_results=10)
        assert _ranked(result) == [("red-dress", 1.0)]
        assert result["total_hits"] == 1

    def test_ranked_and_scored_by_the_whole_phrase(self, db, monkeypatch):
        # With the per-term cut wide open, the order is the phrase search's.
        monkeypatch.setenv("AND_TERM_FRACTION", "1.0")
        result = db.search("red dress", limit=10, max_results=10)
        phrase = db.search('"red dress"', limit=10, max_results=10)
        assert _ranked(result) == _ranked(phrase)

    def test_quoted_phrase_is_one_term(self, db):
        result = db.search('"red dress"', limit=10, max_results=10)
        assert _ranked(result)[0] == ("red-dress", 1.0)
        assert result["total_hits"] == 4
        assert result["terms"] == ["red dress"]

    def test_metadata_filters_scope_the_per_term_cut(self, db):
        # Two products are dresses, so each term keeps its nearest one of them:
        # red-dress for 'red', blue-dress for 'dress' — nothing is near both. Counted
        # against the whole catalogue instead, each term would keep two, i.e. both
        # dresses, and the cut would do nothing.
        db.collection.update(ids=["red-dress", "blue-dress"],
                             metadatas=[{"name": "red-dress", "category": "dress"},
                                        {"name": "blue-dress", "category": "dress"}])
        result = db.search("red dress", categories=["dress"], limit=10, max_results=10)
        assert _ranked(result) == []

    def test_terms_are_translated_one_by_one(self, db):
        result = db.search("赤　ドレス", limit=10, max_results=10)
        assert _ranked(result) == [("red-dress", 1.0)]
        assert result["translated_from"] == "赤　ドレス"
        assert result["query"] == "red dress"
        assert result["terms"] == ["red", "dress"]

    def test_untranslated_query_reports_no_translation(self, db):
        result = db.search("red dress", limit=10, max_results=10)
        assert result["translated_from"] is None

    def test_source_terms_line_up_with_the_embedded_ones(self, db):
        result = db.search("赤　dress", limit=10, max_results=10)
        assert result["source_terms"] == ["赤", "dress"]
        assert result["terms"] == ["red", "dress"]
