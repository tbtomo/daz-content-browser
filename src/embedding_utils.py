import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "BAAI/bge-large-en-v1.5"

# Per-model conventions that change the vectors themselves. Getting these wrong does not
# raise — it silently produces a worse index — so they are pinned here rather than left
# to the caller. Anything not listed keeps the historical behaviour (mean pooling, no
# prefixes), which is what the existing bge-large-en-v1.5 index was built with.
#
#   pooling         'mean' (attention-weighted average) or 'cls' (first token)
#   query_prefix    prepended to search queries      (generate_embeddings(is_query=True))
#   passage_prefix  prepended to indexed documents   (generate_embeddings(is_query=False))
#
# Changing any of these for a model already in ChromaDB invalidates that index —
# queries and documents must be embedded the same way. Re-embed after changing them.
_MODEL_CONVENTIONS = {
    # XLM-RoBERTa based, multilingual. Dense retrieval uses the CLS token and needs
    # no instruction prefix — a Japanese query matches an English passage directly.
    "bge-m3": {"pooling": "cls"},
    # E5 family is mean-pooled but *requires* the asymmetric prefixes; without them
    # retrieval quality collapses.
    "multilingual-e5-large": {"pooling": "mean", "query_prefix": "query: ",
                              "passage_prefix": "passage: "},
    "multilingual-e5-base":  {"pooling": "mean", "query_prefix": "query: ",
                              "passage_prefix": "passage: "},
    "multilingual-e5-small": {"pooling": "mean", "query_prefix": "query: ",
                              "passage_prefix": "passage: "},
}

_config = None
_session = None
_tokenizer = None
_profile_logged = False
_embed_call_count = 0


def _resolve_config() -> dict:
    """Read the model settings from the environment, once, on first use.

    Deliberately *not* evaluated at import time: `main.py` sets EMBEDDING_MODEL_ID from
    its `--model` flag after this module has already been imported (via
    managers.managers), so an import-time constant would freeze the default and make the
    flag a silent no-op.
    """
    global _config
    if _config is not None:
        return _config

    model_id = os.getenv("EMBEDDING_MODEL_ID", DEFAULT_MODEL_ID)
    slug = model_id.split("/")[-1]           # e.g. "bge-large-en-v1.5"
    env_dir = os.getenv("EMBEDDING_MODEL_DIR", "")
    conventions = _MODEL_CONVENTIONS.get(slug, {})

    _config = {
        "model_id": model_id,
        "slug": slug,
        "model_dir": Path(env_dir) if env_dir else (Path(__file__).parent.parent / "models" / slug),
        "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        "max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
        "profile_providers": os.getenv("EMBEDDING_PROFILE_PROVIDERS", "") == "1",
        "pooling": os.getenv("EMBEDDING_POOLING", "") or conventions.get("pooling", "mean"),
        "query_prefix": os.getenv("EMBEDDING_QUERY_PREFIX", conventions.get("query_prefix", "")),
        "passage_prefix": os.getenv("EMBEDDING_PASSAGE_PREFIX", conventions.get("passage_prefix", "")),
    }
    if _config["pooling"] not in ("mean", "cls"):
        raise ValueError(
            f"EMBEDDING_POOLING must be 'mean' or 'cls', got {_config['pooling']!r}"
        )
    return _config


def reset_embedding_model() -> None:
    """Drop the cached config, session and tokenizer. Used by tests and model switching."""
    global _config, _session, _tokenizer, _profile_logged, _embed_call_count
    _config = None
    _session = None
    _tokenizer = None
    _profile_logged = False
    _embed_call_count = 0


def _export_model():
    """Download the configured model from HuggingFace, export to ONNX, and cache locally.

    Uses optimum + transformers — only called from export_model.py, never at server runtime.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    cfg = _resolve_config()
    model_dir = cfg["model_dir"]

    logger.info(f"[embedding] First run — exporting {cfg['model_id']} to ONNX. This may take a few minutes.")
    model_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(cfg["model_id"], export=True)
    model.save_pretrained(str(model_dir))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.save_pretrained(str(model_dir))

    logger.info(f"[embedding] Model exported and saved to {model_dir}")


def _load_from_cache():
    """Load the ONNX session and tokenizer directly — no torch or transformers required."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    cfg = _resolve_config()
    model_dir = cfg["model_dir"]

    logger.info(f"[embedding] Loading ONNX model from {model_dir}")

    available = ort.get_available_providers()
    providers = [p for p in ["DmlExecutionProvider", "CPUExecutionProvider"] if p in available]

    sess_options = ort.SessionOptions()
    if cfg["profile_providers"]:
        # Diagnostic only (EMBEDDING_PROFILE_PROVIDERS=1): session.get_providers() reports
        # which providers are *available* to a session, not which one actually executes
        # each node. ORT's own profiler is the authoritative source for that.
        sess_options.enable_profiling = True

    session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=sess_options, providers=providers)

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=cfg["max_length"])
    tokenizer.enable_padding()

    # session.get_providers() reflects what ORT actually selected, which can silently
    # differ from `providers` above (e.g. DmlExecutionProvider requested but the
    # installed onnxruntime package doesn't include it, so it falls back to CPU).
    logger.info(f"[embedding] ONNX model loaded. Available providers: {available}; "
                f"active providers: {session.get_providers()}")
    logger.info(f"[embedding] model={cfg['model_id']} pooling={cfg['pooling']} "
                f"max_length={cfg['max_length']} "
                f"inputs={[i.name for i in session.get_inputs()]}")
    return session, tokenizer


def load_embedding_model():
    """Load (and if necessary export) the model. Intended to be called at server startup."""
    import sys
    global _session, _tokenizer

    cfg = _resolve_config()
    onnx_path = cfg["model_dir"] / "model.onnx"
    if not onnx_path.exists():
        if getattr(sys, 'frozen', False):
            raise RuntimeError(
                f"ONNX model not found at '{cfg['model_dir']}'. "
                f"Download and export the model with the export tool before switching models."
            )
        _export_model()

    _session, _tokenizer = _load_from_cache()
    return _session, _tokenizer


def get_embedding_model():
    """Return the cached session + tokenizer, raising RuntimeError if the model is unavailable."""
    global _session, _tokenizer
    if _session is None:
        load_embedding_model()
    if _session is None:
        raise RuntimeError(
            f"Embedding model is not loaded. "
            f"Ensure the ONNX model files are present at '{_resolve_config()['model_dir']}'."
        )
    return _session, _tokenizer


def _l2_normalise(pooled: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.maximum(norms, 1e-9)).astype(np.float32)


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token embeddings weighted by the attention mask, then L2-normalise."""
    mask_exp = np.expand_dims(attention_mask, -1).astype(np.float32)  # (batch, seq, 1)
    pooled = (last_hidden_state * mask_exp).sum(axis=1) / mask_exp.sum(axis=1).clip(min=1e-9)
    return _l2_normalise(pooled)


def _cls_pool(last_hidden_state: np.ndarray) -> np.ndarray:
    """Take the first ([CLS]) token, then L2-normalise. What bge-m3 dense retrieval uses."""
    return _l2_normalise(last_hidden_state[:, 0])


def _build_feed(session, encoded) -> dict:
    """Feed only the inputs this ONNX graph actually declares.

    BERT-family exports (bge-large-en-v1.5) take token_type_ids; XLM-RoBERTa-family ones
    (bge-m3, multilingual-e5-*) do not, and passing it raises
    `InvalidArgument: Invalid Feed Input Name: token_type_ids`. Ask the graph instead of
    assuming, so switching model families is a config change rather than a code change.
    """
    available = {
        "input_ids":      lambda: np.array([e.ids for e in encoded], dtype=np.int64),
        "attention_mask": lambda: np.array([e.attention_mask for e in encoded], dtype=np.int64),
        "token_type_ids": lambda: np.array([e.type_ids for e in encoded], dtype=np.int64),
    }
    wanted = [i.name for i in session.get_inputs()]
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise RuntimeError(
            f"[embedding] ONNX model expects input(s) this code cannot supply: {unknown}. "
            f"Known inputs: {sorted(available)}."
        )
    return {name: available[name]() for name in wanted}


def _log_provider_profile(session) -> None:
    """One-shot diagnostic: end ORT profiling and log node count/duration per provider.

    Only runs when EMBEDDING_PROFILE_PROVIDERS=1 is set, and only after the
    second inference call (the first call includes one-time graph compilation
    that would otherwise dominate the numbers).
    """
    import json
    global _profile_logged
    _profile_logged = True
    try:
        profile_path = session.end_profiling()
        with open(profile_path) as f:
            events = json.load(f)
        totals = {}
        counts = {}
        for e in events:
            provider = e.get("args", {}).get("provider") if e.get("cat") == "Node" else None
            if provider:
                totals[provider] = totals.get(provider, 0) + e.get("dur", 0)
                counts[provider] = counts.get(provider, 0) + 1
        summary = ", ".join(f"{p}: {counts[p]} nodes/{totals[p]}us" for p in totals)
        logger.info(f"[embedding] Provider profile (one-shot): {summary}")
    except Exception:
        logger.exception("[embedding] Failed to read provider profile")


def generate_embeddings(texts, is_query: bool = False) -> np.ndarray:
    """Tokenise, run ONNX inference, pool, and L2-normalise.

    Processes texts in sub-batches of EMBEDDING_BATCH_SIZE (default 32) to
    keep peak memory reasonable on CPU.

    `is_query` selects the instruction prefix for models that use asymmetric
    query/passage encoding (the E5 family). Models that need no prefix — including
    bge-large-en-v1.5 and bge-m3 — ignore it.

    Returns float32 ndarray of shape (dim,) for a single string, or (N, dim) for a list.
    The dimension follows the model: 1024 for both bge-large-en-v1.5 and bge-m3.
    """
    session, tokenizer = get_embedding_model()
    cfg = _resolve_config()

    single = isinstance(texts, str)
    if single:
        texts = [texts]

    prefix = cfg["query_prefix"] if is_query else cfg["passage_prefix"]
    if prefix:
        texts = [prefix + t for t in texts]

    logger.debug(f"[embedding] Generating embeddings for {len(texts)} text(s)")

    batch_size = cfg["batch_size"]
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size
    chunks = []
    for idx, start in enumerate(range(0, total, batch_size)):
        batch = texts[start: start + batch_size]
        logger.info(f"[embedding] Sub-batch {idx + 1}/{n_batches} ({len(batch)} texts)")

        encoded = tokenizer.encode_batch(batch)
        feed = _build_feed(session, encoded)

        outputs = session.run(None, feed)

        if cfg["pooling"] == "cls":
            chunks.append(_cls_pool(outputs[0]))
        else:
            chunks.append(_mean_pool(outputs[0], feed["attention_mask"]))

    if cfg["profile_providers"] and not _profile_logged:
        global _embed_call_count
        _embed_call_count += 1
        # Skip the first call: it includes one-time graph compilation that would
        # otherwise dominate the per-provider timing.
        if _embed_call_count >= 2:
            _log_provider_profile(session)

    embeddings = np.concatenate(chunks, axis=0)
    return embeddings[0] if single else embeddings
