"""Tests for the Japanese-to-English query translator.

The parts worth pinning are the ones that fail silently: the language test (an
English query must never reach the model), the fallbacks (a missing model must
degrade to the raw query, not raise), and the hand-rolled Marian tokenizer, which
would still produce *some* text if it were subtly wrong.
"""
import json
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
def clean_state(monkeypatch, tmp_path):
    # Point away from the shipped glossary by default. Terms get added to it over
    # time, and a new entry must not quietly stop a test from exercising the model.
    monkeypatch.setenv("QUERY_GLOSSARY_PATH", str(tmp_path / "no-glossary.json"))
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

    def test_auto_follows_whether_a_model_was_downloaded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_TRANSLATION", "auto")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        with patch.object(qt, "model_dir_for", return_value=tmp_path):
            assert qt.is_enabled() is False
            (tmp_path / "encoder_model.onnx").write_bytes(b"")
            assert qt.is_enabled() is True

    def test_auto_accepts_the_other_backends_model(self, monkeypatch, tmp_path):
        """Either backend counts — the loader falls back between them."""
        monkeypatch.setenv("QUERY_TRANSLATION", "auto")
        monkeypatch.setenv("QUERY_TRANSLATION_BACKEND", "ctranslate2")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        other = tmp_path / "other"
        other.mkdir()
        (other / "encoder_model.onnx").write_bytes(b"")
        with patch.object(qt, "model_dir_for", return_value=other):
            assert qt.is_enabled() is True


class TestPassThrough:
    """A translator that is missing or broken must cost quality, never a search."""

    def test_english_never_touches_the_model(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        with patch.object(qt, "_load", side_effect=AssertionError("model was loaded")):
            assert qt.translate_query("office") == ("office", None)

    def test_missing_models_return_the_query_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("QUERY_GLOSSARY_PATH", str(tmp_path / "none.json"))
        with patch.object(qt, "model_dir_for", return_value=tmp_path):
            assert qt.translate_query("和室") == ("和室", None)

    def test_a_failing_decode_returns_the_query_unchanged(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")

        def boom(_text):
            raise RuntimeError("the model blew up")

        with patch.object(qt, "_load", return_value={"run": boom}):
            assert qt.translate_query("和室") == ("和室", None)

    def test_empty_output_falls_back_to_the_query(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        with patch.object(qt, "_load", return_value={"run": lambda _t: "   "}):
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
        ("森の風景", ["forest"]),
        ("学校の教室", ["school", "classroom"]),
        ("畳の部屋", ["tatami"]),
    ])
    def test_translates_to_english(self, monkeypatch, japanese, expected_words):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_GLOSSARY_PATH", "/nonexistent")
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


CT2_DIR = Path(__file__).parent.parent / "models" / "sugoi-v4-ja-en-ctranslate2"
needs_ct2 = pytest.mark.skipif(
    not (CT2_DIR / "model.bin").is_file(),
    reason="ctranslate2 model not downloaded "
           "(python export_translation_model.py --backend ctranslate2)",
)


class TestGlossary:
    """Terms the model gets wrong are answered from a table instead."""

    def _write(self, tmp_path, terms):
        path = tmp_path / "glossary.json"
        path.write_text(json.dumps({"terms": terms}), encoding="utf-8")
        return path

    def test_a_whole_query_hit_skips_the_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_GLOSSARY_PATH",
                           str(self._write(tmp_path, {"縄": "rope"})))
        qt.reset_cache()
        with patch.object(qt, "_load", side_effect=AssertionError("model was loaded")):
            assert qt.translate_query("縄") == ("rope", "縄")

    def test_surrounding_text_is_not_rewritten(self, monkeypatch, tmp_path):
        """沖縄 must not become 'Okirope' — matching is on the whole query only."""
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_GLOSSARY_PATH",
                           str(self._write(tmp_path, {"縄": "rope"})))
        qt.reset_cache()
        with patch.object(qt, "_load", return_value={"run": lambda t: "Okinawa scenery"}):
            english, source = qt.translate_query("沖縄の風景")
        assert english == "Okinawa scenery"
        assert source == "沖縄の風景"

    def test_a_missing_file_is_not_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_GLOSSARY_PATH", str(tmp_path / "nope.json"))
        qt.reset_cache()
        assert qt.load_glossary() == {}

    def test_malformed_json_is_not_an_error(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not json", encoding="utf-8")
        monkeypatch.setenv("QUERY_GLOSSARY_PATH", str(path))
        qt.reset_cache()
        assert qt.load_glossary() == {}

    def test_the_shipped_glossary_parses(self):
        """Read the real file directly — clean_state points the loader elsewhere."""
        path = Path(__file__).parent.parent / "query_glossary.ja-en.json"
        terms = json.loads(path.read_text(encoding="utf-8"))["terms"]
        assert terms, "the shipped glossary should not be empty"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in terms.items())
        assert all(qt.contains_japanese(k) for k in terms), \
            "glossary keys are Japanese queries"
        assert not any(qt.contains_japanese(v) for v in terms.values()), \
            "glossary values are the English wording the index uses"


class TestBackendSelection:
    def test_unknown_backend_falls_back_to_onnx(self, monkeypatch):
        monkeypatch.setenv("QUERY_TRANSLATION_BACKEND", "nonsense")
        assert qt._config()["backend"] == "onnx"

    def test_backend_picks_its_own_default_model(self, monkeypatch):
        monkeypatch.delenv("QUERY_TRANSLATION_MODEL_ID", raising=False)
        monkeypatch.delenv("QUERY_TRANSLATION_MODEL_DIR", raising=False)
        monkeypatch.setenv("QUERY_TRANSLATION_BACKEND", "ctranslate2")
        assert qt._config()["model_id"] == qt.DEFAULT_CT2_MODEL_ID
        monkeypatch.setenv("QUERY_TRANSLATION_BACKEND", "onnx")
        assert qt._config()["model_id"] == qt.DEFAULT_MODEL_ID

    def test_an_unavailable_backend_falls_back_to_the_other(self, monkeypatch, tmp_path):
        """A configured backend with no model must not disable translation."""
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_TRANSLATION_BACKEND", "ctranslate2")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        with patch.dict(qt._LOADERS, {"onnx": lambda d: {"run": lambda t: "fallback"}}):
            assert qt.translate_query("和室") == ("fallback", "和室")
            assert qt.active_backend() == "onnx"

    def test_neither_backend_available_passes_the_query_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_TRANSLATION_MODEL_DIR", str(tmp_path))
        with patch.dict(qt._LOADERS, {
            "onnx": lambda d: (_ for _ in ()).throw(FileNotFoundError("no onnx")),
            "ctranslate2": lambda d: (_ for _ in ()).throw(FileNotFoundError("no ct2")),
        }):
            assert qt.translate_query("和室") == ("和室", None)


@needs_ct2
class TestCTranslate2Backend:
    @pytest.mark.parametrize("japanese, expected", [
        ("緊縛", "bondage"),
        ("鞭", "whip"),
        ("セックス", "sex"),
        ("拘束", "restraint"),
    ])
    def test_translates_the_vocabulary_fugumt_misses(self, monkeypatch, japanese, expected):
        monkeypatch.setenv("QUERY_TRANSLATION", "on")
        monkeypatch.setenv("QUERY_TRANSLATION_BACKEND", "ctranslate2")
        monkeypatch.delenv("QUERY_TRANSLATION_MODEL_DIR", raising=False)
        monkeypatch.setenv("QUERY_GLOSSARY_PATH", "/nonexistent")
        english, _ = qt.translate_query(japanese)
        assert expected in english.lower(), english
