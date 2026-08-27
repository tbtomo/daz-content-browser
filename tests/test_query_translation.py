"""Tests for the Japanese-to-English query translator.

The parts worth pinning are the ones that fail silently: the language test (an
English query must never reach the model), the fallbacks (a missing model must
degrade to the raw query, not raise), and the hand-rolled Marian tokenizer, which
would still produce *some* text if it were subtly wrong.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import query_translation as qt  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent / "models" / "fugumt-ja-en"
needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "encoder_model.onnx").is_file(),
    reason="translation model not exported (python export_translation_model.py)",
)


@pytest.fixture(autouse=True)
def clean_state():
    qt.reset_cache()
    yield
    qt.reset_cache()


class TestLanguageDetection:
    @pytest.mark.parametrize("text", [
        "和室の背景",          # kanji
        "ロングヘア",          # katakana
        "ながいかみ",          # hiragana
        "8K の背景",           # mixed with latin
        "ｱｲｺ",                 # halfwidth katakana
    ])
    def test_japanese_is_detected(self, text):
        assert qt.contains_japanese(text) is True

    @pytest.mark.parametrize("text", [
        "office", "long hair female character", "G8F dForce", "", "3D 8K v4.2",
    ])
    def test_latin_is_not(self, text):
        assert qt.contains_japanese(text) is False


class TestEnablement:
    def test_off_disables_even_with_a_model_present(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "off")
        assert qt.is_enabled() is False

    def test_auto_follows_whether_the_model_was_exported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_TRANSLATION", "auto")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        assert qt.is_enabled() is False
        (tmp_path / "encoder_model.onnx").write_bytes(b"")
        assert qt.is_enabled() is True


class TestPassThrough:
    """A translator that is missing or broken must cost quality, never a search."""

    def test_english_never_touches_the_model(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        with patch.object(qt, "_load", side_effect=AssertionError("model was loaded")):
            assert qt.translate_query("office") == ("office", None)

    def test_missing_model_returns_the_query_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        assert qt.translate_query("和室") == ("和室", None)

    def test_a_failing_decode_returns_the_query_unchanged(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        with patch.object(qt, "_load", return_value={"tokenizer": None}), \
             patch.object(qt, "_greedy_decode", side_effect=RuntimeError("onnx blew up")):
            assert qt.translate_query("和室") == ("和室", None)

    def test_empty_output_falls_back_to_the_query(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        with patch.object(qt, "_load", return_value={"tokenizer": None}), \
             patch.object(qt, "_greedy_decode", return_value="   "):
            assert qt.translate_query("和室") == ("和室", None)


@pytest.fixture(scope="module")
def pieces():
    return qt.MarianPieces(MODEL_DIR)


@needs_model
class TestMarianPieces:
    """The tokenizer is reimplemented on sentencepiece to keep transformers out of
    the packaged build, so it has to agree with the reference exactly."""

    @pytest.mark.parametrize("text", [
        "和室の背景", "和室", "ロングヘアの女性キャラ", "中世の鎧を着た戦士",
        "森の風景", "学校の教室", "畳の部屋", "障子のある旅館の客室",
        "ゴシック調のドレスを着た少女", "雪山の背景 8K",
    ])
    def test_ids_match_the_transformers_tokenizer(self, pieces, text):
        transformers = pytest.importorskip("transformers")
        reference = transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))
        assert pieces.encode(text) == reference(text)["input_ids"]

    def test_encoding_ends_with_eos(self, pieces):
        assert pieces.encode("和室")[-1] == pieces.eos_id

    def test_truncation_keeps_room_for_eos(self, pieces):
        ids = pieces.encode("和室の背景 " * 400, max_length=32)
        assert len(ids) == 32
        assert ids[-1] == pieces.eos_id

    def test_decode_drops_control_tokens(self, pieces):
        ids = pieces.encode("森の風景")
        assert "</s>" not in pieces.decode(ids)


@needs_model
class TestTranslation:
    @pytest.mark.parametrize("japanese, expected_words", [
        ("和室", ["japanese"]),
        ("森の風景", ["forest"]),
        ("学校の教室", ["school", "classroom"]),
        ("畳の部屋", ["tatami"]),
    ])
    def test_translates_to_english(self, monkeypatch, japanese, expected_words):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        english, source = qt.translate_query(japanese)
        assert source == japanese
        assert not qt.contains_japanese(english)
        lowered = english.lower()
        for word in expected_words:
            assert word in lowered, f"{english!r} lacks {word!r}"

    def test_repeated_queries_are_cached(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        qt.translate_query("和室")
        before = qt._translate_cached.cache_info()
        qt.translate_query("和室")
        after = qt._translate_cached.cache_info()
        assert after.hits == before.hits + 1
