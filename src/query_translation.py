"""Translates Japanese search queries to English before they are embedded.

The index is English — product names and descriptions scraped from the DAZ store,
path words for filesystem products. A Japanese query has no lexical overlap to lean
on, so it depends entirely on the embedding model bridging the two languages, and
that bridge is the weak link: '和室の背景' reproduces only 3 of the 10 results its
English phrasing finds (see Handover.pyn @xl-bench). Swapping in a different
multilingual model was measured and made other things worse (@e5-trial).

Translating the query instead lets the existing model do the one thing it is good
at — an English query against English text.

Two backends, because one model does not cover both registers:

* ``onnx`` — ``staka/fugumt-ja-en`` run on onnxruntime. The default. Excellent on
  everyday language (9.0/10 on @xl-bench) and, at ~360 MB, cheap.
* ``ctranslate2`` — ``entai2965/sugoi-v4-ja-en-ctranslate2``. General-purpose
  ja-en corpora are filtered for adult content, which leaves fugumt guessing at a
  whole domain: 緊縛 came out 'tight-knit', 鞭 'sputum', セックス 'sleek', each
  sending the search somewhere unrelated. Sugoi was trained with that vocabulary
  and returns Bondage / Whip / Sex. Costs 1.1 GB, ~130 ms a query, and carries a
  research-only licence — hence opt-in rather than default.

Runtime constraints this module works within:

* **No torch, and no transformers either.** ``vab.spec`` excludes both from the
  packaged build — they exist for the one-time ONNX export and nothing else. So the
  ONNX graph is driven directly with onnxruntime and numpy (a hand-written greedy
  decode loop, since ``transformers.generate()`` is torch-only), and the Marian
  tokenizer is reproduced on top of ``sentencepiece`` plus the exported
  ``vocab.json``. ``embedding_utils`` keeps the same split for the same reason; it
  can use the fast ``tokenizers`` library because its models ship a
  ``tokenizer.json``, which Marian does not. The piece ids this produces are
  identical to ``MarianTokenizer``'s — ``tests/test_query_translation.py`` pins
  that. CTranslate2 is likewise a self-contained C++ wheel with no torch.
* **Optional, at every level.** A missing backend falls back to the other one, and
  a missing translator falls back to the raw query. Losing the translator costs
  search quality; it must never fail a search.
* **Latin queries cost nothing.** Text with no Japanese characters skips the model
  entirely, so an English query pays neither the load nor the latency.

Get the models with::

    python export_translation_model.py                        # onnx, ~360 MB
    python export_translation_model.py --backend ctranslate2  # sugoi, ~1.1 GB
"""

import functools
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Same as embedding_utils: the settings below live in .env, and this module is
# importable on its own (tests, the benchmark scripts) without a caller having
# arranged for it.
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "staka/fugumt-ja-en"
DEFAULT_CT2_MODEL_ID = "entai2965/sugoi-v4-ja-en-ctranslate2"
BACKENDS = ("onnx", "ctranslate2")

# Hiragana, katakana, CJK ideographs (incl. extension A) and halfwidth katakana.
# Latin and digits are deliberately not enough on their own — an English query must
# not pay for the translator. ui/dist uses the same class to decide whether a query
# is long enough to send.
_JAPANESE_RE = re.compile(
    r"[぀-ゟ゠-ヿ㐀-䶿一-鿿ｦ-ﾟ]"
)

# Queries are short noun phrases; 64 new tokens is far more than any of them needs,
# and it bounds the worst case should the model fail to emit EOS.
MAX_NEW_TOKENS = 64

_state = None
_glossary = None
_unavailable_logged = False


def contains_japanese(text: str) -> bool:
    """True when the text holds at least one Japanese character."""
    return bool(text) and _JAPANESE_RE.search(text) is not None


def _config() -> dict:
    backend = (os.getenv("QUERY_TRANSLATION_BACKEND", "onnx") or "onnx").lower()
    if backend not in BACKENDS:
        logger.warning(f"[translation] unknown backend {backend!r}, using 'onnx'")
        backend = "onnx"
    default_id = DEFAULT_CT2_MODEL_ID if backend == "ctranslate2" else DEFAULT_MODEL_ID
    model_id = os.getenv("QUERY_TRANSLATION_MODEL_ID", default_id)
    env_dir = os.getenv("QUERY_TRANSLATION_MODEL_DIR", "")
    return {
        "backend": backend,
        "model_id": model_id,
        "mode": (os.getenv("QUERY_TRANSLATION", "auto") or "auto").lower(),
        "model_dir": (Path(env_dir) if env_dir else model_dir_for(model_id)),
    }


def model_dir_for(model_id: str) -> Path:
    """Local directory a model ID is cached in. Shared with the export script."""
    return Path(__file__).parent.parent / "models" / model_id.split("/")[-1]


def _marker(backend: str) -> str:
    """The file whose presence means 'this backend's model is downloaded'."""
    return "model.bin" if backend == "ctranslate2" else "encoder_model.onnx"


def is_enabled() -> bool:
    """True when a Japanese query would actually be translated.

    'auto' (the default) turns the translator on precisely when a model has been
    downloaded, so a clone that never ran export_translation_model.py keeps the
    previous behaviour rather than erroring. Either backend's model counts, since
    the loader falls back between them.
    """
    cfg = _config()
    if cfg["mode"] == "off":
        return False
    if cfg["mode"] == "on":
        return True
    if (cfg["model_dir"] / _marker(cfg["backend"])).is_file():
        return True
    fallback = "onnx" if cfg["backend"] == "ctranslate2" else "ctranslate2"
    default_id = DEFAULT_CT2_MODEL_ID if fallback == "ctranslate2" else DEFAULT_MODEL_ID
    return (model_dir_for(default_id) / _marker(fallback)).is_file()


# ─── Glossary ──────────────────────────────────────────────────────────────────

def _glossary_path() -> Path:
    override = os.getenv("QUERY_GLOSSARY_PATH", "")
    if override:
        return Path(override)
    return Path(__file__).parent.parent / "query_glossary.ja-en.json"


def load_glossary() -> dict:
    """Terms a translator gets wrong, mapped straight to the English the index uses.

    Every model has holes. Sugoi romanises rather than translates a few words —
    縄 becomes 'Nawa', 猿ぐつわ 'Sarugutsuwa' — which searches for nothing. A
    lookup table fixes those deterministically, and the user can extend it without
    touching code.
    """
    global _glossary
    if _glossary is not None:
        return _glossary
    path = _glossary_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _glossary = {k.strip(): v for k, v in raw.get("terms", {}).items() if k.strip()}
        logger.info(f"[translation] glossary: {len(_glossary)} term(s) from {path}")
    except FileNotFoundError:
        _glossary = {}
    except (OSError, ValueError, AttributeError) as e:
        logger.warning(f"[translation] could not read glossary {path}: {e}")
        _glossary = {}
    return _glossary


def _glossary_hit(text: str):
    """Looks the whole query up in the glossary.

    Deliberately an exact match on the trimmed query rather than substring
    replacement: single kanji recur inside unrelated words — rewriting 縄 wherever
    it appears would turn 沖縄 (Okinawa) into 'rope' — and a search query is
    usually the bare term anyway.
    """
    return load_glossary().get(text.strip())


# ─── Marian / ONNX backend ─────────────────────────────────────────────────────

class MarianPieces:
    """The Marian tokenizer, rebuilt on sentencepiece so runtime needs no transformers.

    Marian is one of the few architectures transformers has no fast tokenizer for, so
    there is no ``tokenizer.json`` to hand to the ``tokenizers`` library the way
    ``embedding_utils`` does. The pieces themselves are plain sentencepiece, though,
    and the exported ``vocab.json`` maps them to ids — encoding is the two composed,
    with ``</s>`` appended.

    Source and target use *separate* sentencepiece models (``source.spm`` /
    ``target.spm``) over one shared vocabulary, which is why decoding cannot simply
    reuse the encoder's processor.
    """

    def __init__(self, directory: Path):
        import sentencepiece as spm

        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        self._vocab = vocab
        self._inverse = {i: piece for piece, i in vocab.items()}
        self._unk = vocab.get("<unk>", 1)
        self._eos = vocab.get("</s>", 0)
        self._source = spm.SentencePieceProcessor(model_file=str(directory / "source.spm"))
        self._target = spm.SentencePieceProcessor(model_file=str(directory / "target.spm"))

    @property
    def pad_id(self) -> int:
        return self._vocab.get("<pad>", 32000)

    @property
    def eos_id(self) -> int:
        return self._eos

    def encode(self, text: str, max_length: int = 512) -> list:
        pieces = self._source.encode(text, out_type=str)
        ids = [self._vocab.get(p, self._unk) for p in pieces][: max_length - 1]
        return ids + [self._eos]

    def decode(self, ids) -> str:
        pieces = [self._inverse[i] for i in ids
                  if i in self._inverse and self._inverse[i] not in ("<pad>", "</s>", "<unk>")]
        return self._target.decode(pieces)


def _token_id(directory: Path, key: str, fallback):
    """Reads a special token id from generation_config.json, then config.json."""
    for name in ("generation_config.json", "config.json"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
        except (OSError, ValueError):
            continue
        if isinstance(value, int):
            return value
    return fallback


def _load_onnx(directory: Path) -> dict:
    import onnxruntime as ort

    encoder_path = directory / "encoder_model.onnx"
    decoder_path = directory / "decoder_model.onnx"
    if not encoder_path.is_file() or not decoder_path.is_file():
        raise FileNotFoundError(
            f"{directory} holds no exported ONNX translation model — "
            "run: python export_translation_model.py"
        )

    # CPU only. The graph is small, and the embedding model already holds the GPU
    # provider; two sessions contending for one device costs more than it saves on
    # a query of a dozen tokens.
    providers = ["CPUExecutionProvider"]
    tokenizer = MarianPieces(directory)
    encoder = ort.InferenceSession(str(encoder_path), providers=providers)
    decoder = ort.InferenceSession(str(decoder_path), providers=providers)
    start_id = _token_id(directory, "decoder_start_token_id", tokenizer.pad_id)
    eos_id = _token_id(directory, "eos_token_id", tokenizer.eos_id)

    def run(text: str) -> str:
        return _greedy_decode(tokenizer, encoder, decoder, start_id, eos_id, text)

    return {"run": run}


def _greedy_decode(tokenizer, encoder, decoder, start_id, eos_id, text: str) -> str:
    """Runs the encoder once, then the decoder a token at a time, taking the argmax.

    transformers' generate() would do this, but it builds torch tensors. The model
    config asks for 12 beams; greedy is used instead because a search query is a
    short noun phrase where beam search buys little, and it keeps this to two ONNX
    sessions and one loop rather than a beam manager.
    """
    input_ids = np.array([tokenizer.encode(text)], dtype=np.int64)
    # One sequence, no padding, so every position is attended to.
    attention_mask = np.ones_like(input_ids)

    hidden = encoder.run(["last_hidden_state"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0]

    generated = [start_id]
    for _ in range(MAX_NEW_TOKENS):
        logits = decoder.run(["logits"], {
            "input_ids": np.array([generated], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": attention_mask,
        })[0]
        step = logits[0, -1].astype(np.float64)
        # The start/pad token is a real vocabulary entry the model must never emit
        # mid-sentence; Marian's own generation config bans it the same way.
        step[start_id] = -np.inf
        next_id = int(np.argmax(step))
        if next_id == eos_id:
            break
        generated.append(next_id)

    return tokenizer.decode(generated[1:]).strip()


# ─── CTranslate2 backend ───────────────────────────────────────────────────────

def _load_ctranslate2(directory: Path) -> dict:
    import ctranslate2
    import sentencepiece as spm

    if not (directory / "model.bin").is_file():
        raise FileNotFoundError(
            f"{directory} holds no CTranslate2 model — run: "
            "python export_translation_model.py --backend ctranslate2"
        )

    # Sugoi ships its sentencepiece models under spm/; other CTranslate2 conversions
    # put source.spm / target.spm at the top level.
    spm_dir = directory / "spm" if (directory / "spm").is_dir() else directory
    source_spm = _first_existing(spm_dir, ("spm.ja.nopretok.model", "source.spm"))
    target_spm = _first_existing(spm_dir, ("spm.en.nopretok.model", "target.spm"))

    translator = ctranslate2.Translator(str(directory), device="cpu")
    source = spm.SentencePieceProcessor(model_file=str(source_spm))
    target = spm.SentencePieceProcessor(model_file=str(target_spm))

    def run(text: str) -> str:
        tokens = source.encode(text, out_type=str)
        # Beam search is affordable here: CTranslate2 is fast enough that a query
        # still lands around 130 ms, well inside a search that takes 2.4 s.
        results = translator.translate_batch([tokens], beam_size=5, max_decoding_length=MAX_NEW_TOKENS)
        return target.decode(results[0].hypotheses[0]).strip()

    return {"run": run}


def _first_existing(directory: Path, names):
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"none of {names} found in {directory}")


# ─── Loading and translation ───────────────────────────────────────────────────

_LOADERS = {"onnx": _load_onnx, "ctranslate2": _load_ctranslate2}


def _load():
    """Loads the configured backend, falling back to the other one. None if neither."""
    global _state, _unavailable_logged
    if _state is not None:
        return _state or None

    cfg = _config()
    order = [cfg["backend"]] + [b for b in BACKENDS if b != cfg["backend"]]
    errors = []
    for backend in order:
        model_id = cfg["model_id"] if backend == cfg["backend"] else (
            DEFAULT_CT2_MODEL_ID if backend == "ctranslate2" else DEFAULT_MODEL_ID)
        directory = cfg["model_dir"] if backend == cfg["backend"] else model_dir_for(model_id)
        try:
            _state = dict(_LOADERS[backend](directory),
                          backend=backend, model_id=model_id)
            if backend != cfg["backend"]:
                logger.warning(
                    f"[translation] backend {cfg['backend']!r} unavailable, "
                    f"fell back to {backend!r}")
            logger.info(f"[translation] loaded {model_id} ({backend}) from {directory}")
            return _state
        except Exception as e:
            errors.append(f"{backend}: {e}")

    if not _unavailable_logged:
        logger.warning("[translation] unavailable, queries pass through untranslated — "
                       + "; ".join(errors))
        _unavailable_logged = True
    _state = {}
    return None


@functools.lru_cache(maxsize=512)
def _translate_cached(text: str) -> str:
    known = _glossary_hit(text)
    if known:
        return known
    state = _load()
    if state is None:
        return text
    try:
        english = state["run"](text)
    except Exception as e:
        logger.warning(f"[translation] failed for {text!r}, using it as-is: {e}")
        return text
    # A blank result means the decode produced nothing usable — searching for it
    # would return the whole catalogue in arbitrary order.
    return english.strip() or text


def translate_query(text: str):
    """Returns (query_to_embed, translated_from) for a raw search query.

    ``translated_from`` is None when the text is used unchanged — not Japanese, the
    translator is off or missing, or translation failed — so callers can report what
    was actually searched without having to re-detect any of that.
    """
    if not text or not contains_japanese(text) or not is_enabled():
        return text, None
    english = _translate_cached(text)
    if english == text:
        return text, None
    logger.info(f"[translation] {text!r} -> {english!r}")
    return english, text


def active_backend() -> str:
    """The backend actually in use, or '' if the translator could not load."""
    state = _load()
    return state.get("backend", "") if state else ""


def reset_cache() -> None:
    """Drops the loaded model, the glossary and the memoised translations. For tests."""
    global _state, _glossary, _unavailable_logged
    _state = None
    _glossary = None
    _unavailable_logged = False
    _translate_cached.cache_clear()
