"""Model-switching logic in embedding_utils — no ONNX model or download required.

These cover the two ways a model swap fails silently or fatally:
  * feeding token_type_ids to a graph that has no such input (bge-m3, multilingual-e5-*)
  * pooling or prefixing a model the way a *different* model wants, which does not raise
    but quietly produces a worse index
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import embedding_utils as eu  # noqa: E402


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Stands in for onnxruntime.InferenceSession — only get_inputs() is exercised."""

    def __init__(self, names):
        self._names = names

    def get_inputs(self):
        return [FakeInput(n) for n in self._names]


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


@pytest.fixture
def encoded():
    return [FakeEncoding([101, 2023, 102]), FakeEncoding([101, 3231, 102])]


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    """Each test resolves the config from scratch, and leaves nothing cached behind."""
    for var in ("EMBEDDING_MODEL_ID", "EMBEDDING_POOLING", "EMBEDDING_QUERY_PREFIX",
                "EMBEDDING_PASSAGE_PREFIX", "EMBEDDING_MAX_LENGTH"):
        monkeypatch.delenv(var, raising=False)
    eu.reset_embedding_model()
    yield
    eu.reset_embedding_model()


# ── ONNX input negotiation ────────────────────────────────────────────────────

def test_bert_family_graph_gets_token_type_ids(encoded):
    feed = eu._build_feed(FakeSession(["input_ids", "attention_mask", "token_type_ids"]), encoded)
    assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}
    assert feed["input_ids"].dtype == np.int64
    assert feed["input_ids"].shape == (2, 3)


def test_xlm_roberta_graph_does_not_get_token_type_ids(encoded):
    """bge-m3 / multilingual-e5-* declare only two inputs.

    Passing token_type_ids anyway is what makes ORT raise
    'InvalidArgument: Invalid Feed Input Name: token_type_ids'.
    """
    feed = eu._build_feed(FakeSession(["input_ids", "attention_mask"]), encoded)
    assert set(feed) == {"input_ids", "attention_mask"}


def test_unknown_graph_input_is_reported_not_ignored(encoded):
    with pytest.raises(RuntimeError, match="position_ids"):
        eu._build_feed(FakeSession(["input_ids", "position_ids"]), encoded)


# ── Per-model conventions ─────────────────────────────────────────────────────

def test_default_model_keeps_historical_behaviour():
    """The existing ChromaDB index was built mean-pooled and unprefixed. Don't move it."""
    cfg = eu._resolve_config()
    assert cfg["model_id"] == "BAAI/bge-large-en-v1.5"
    assert cfg["pooling"] == "mean"
    assert cfg["query_prefix"] == "" and cfg["passage_prefix"] == ""


def test_bge_m3_uses_cls_pooling_and_no_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    cfg = eu._resolve_config()
    assert cfg["pooling"] == "cls"
    assert cfg["query_prefix"] == "" and cfg["passage_prefix"] == ""
    assert cfg["model_dir"].name == "bge-m3"


def test_e5_family_gets_asymmetric_prefixes(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-large")
    cfg = eu._resolve_config()
    assert cfg["pooling"] == "mean"
    assert cfg["query_prefix"] == "query: "
    assert cfg["passage_prefix"] == "passage: "


def test_env_overrides_built_in_convention(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_POOLING", "mean")
    assert eu._resolve_config()["pooling"] == "mean"


def test_invalid_pooling_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_POOLING", "max")
    with pytest.raises(ValueError, match="EMBEDDING_POOLING"):
        eu._resolve_config()


def test_model_id_is_read_lazily_not_at_import(monkeypatch):
    """`main.py --model` sets the env var after this module is already imported."""
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    assert eu._resolve_config()["model_id"] == "BAAI/bge-m3"
    eu.reset_embedding_model()
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-large")
    assert eu._resolve_config()["model_id"] == "intfloat/multilingual-e5-large"


# ── Pooling maths ─────────────────────────────────────────────────────────────

def test_cls_pool_takes_first_token_and_normalises():
    hidden = np.array([[[3.0, 4.0], [99.0, 99.0]]], dtype=np.float32)
    out = eu._cls_pool(hidden)
    assert np.allclose(out, [[0.6, 0.8]])


def test_mean_pool_ignores_padding():
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [1000.0, 1000.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    out = eu._mean_pool(hidden, mask)
    assert np.allclose(out, [[1.0, 0.0]])
