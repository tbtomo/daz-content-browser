"""Translates Japanese search queries to English before they are embedded.

The index is English — product names and descriptions scraped from the DAZ store,
path words for filesystem products. A Japanese query has no lexical overlap to lean
on, so it depends entirely on the embedding model bridging the two languages, and
that bridge is the weak link: '和室の背景' reproduces only 3 of the 10 results its
English phrasing finds (see Handover.pyn @xl-bench). Swapping in a different
multilingual model was measured and made other things worse (@e5-trial).

Translating the query instead lets the existing model do the one thing it is good
at — an English query against English text.

Runtime constraints this module works within:

* **No torch, and no transformers either.** ``vab.spec`` excludes both from the
  packaged build — they exist for the one-time ONNX export and nothing else. So the
  graph is driven directly with onnxruntime and numpy (a hand-written greedy decode
  loop, since ``transformers.generate()`` is torch-only), and the Marian tokenizer
  is reproduced on top of ``sentencepiece`` plus the exported ``vocab.json``.
  ``embedding_utils`` keeps the same split for the same reason; it can use the fast
  ``tokenizers`` library because its models ship a ``tokenizer.json``, which Marian
  does not. The piece ids this produces are identical to
  ``MarianTokenizer``'s — ``tests/test_query_translation.py`` pins that.
* **Optional.** If the model is not exported, or anything at all goes wrong, the
  original query is used unchanged. A missing translator costs search quality; it
  must never fail a search.
* **Latin queries cost nothing.** Text with no Japanese characters skips the model
  entirely, so an English query pays neither the load nor the latency.

Export the model with::

    python export_translation_model.py
"""

import functools
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "staka/fugumt-ja-en"

# Hiragana, katakana, CJK ideographs (incl. extension A) and halfwidth katakana.
# Latin and digits are deliberately not enough on their own — an English query must
# not pay for the translator.
_JAPANESE_RE = re.compile(
    r"[぀-ゟ゠-ヿ㐀-䶿一-鿿ｦ-ﾟ]"
)

# Queries are short noun phrases; 64 new tokens is far more than any of them needs,
# and it bounds the worst case should the model fail to emit EOS.
MAX_NEW_TOKENS = 64

_state = None
_unavailable_logged = False


def contains_japanese(text: str) -> bool:
    """True when the text holds at least one Japanese character."""
    return bool(text) and _JAPANESE_RE.search(text) is not None


def _config() -> dict:
    model_id = os.getenv("QUERY_TRANSLATION_MODEL_ID", DEFAULT_MODEL_ID)
    env_dir = os.getenv("QUERY_TRANSLATION_MODEL_DIR", "")
    slug = model_id.split("/")[-1]
    return {
        "model_id": model_id,
        "mode": (os.getenv("QUERY_TRANSLATION", "auto") or "auto").lower(),
        "model_dir": (Path(env_dir) if env_dir
                      else Path(__file__).parent.parent / "models" / slug),
    }


def is_enabled() -> bool:
    """True when a Japanese query would actually be translated.

    'auto' (the default) turns the translator on precisely when its model has been
    exported, so a clone that never ran export_translation_model.py keeps the
    previous behaviour rather than erroring.
    """
    cfg = _config()
    if cfg["mode"] == "off":
        return False
    if cfg["mode"] == "on":
        return True
    return (cfg["model_dir"] / "encoder_model.onnx").is_file()


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


def _load():
    """Loads the tokenizer and both ONNX sessions. Returns None when unavailable."""
    global _state, _unavailable_logged
    if _state is not None:
        return _state or None

    cfg = _config()
    directory = cfg["model_dir"]
    try:
        import onnxruntime as ort

        encoder_path = directory / "encoder_model.onnx"
        decoder_path = directory / "decoder_model.onnx"
        if not encoder_path.is_file() or not decoder_path.is_file():
            raise FileNotFoundError(
                f"{directory} holds no exported translation model — "
                "run: python export_translation_model.py"
            )

        # CPU only. The graph is small, and the embedding model already holds the
        # GPU provider; two sessions contending for one device costs more than it
        # saves on a query of a dozen tokens.
        providers = ["CPUExecutionProvider"]
        tokenizer = MarianPieces(directory)
        _state = {
            "tokenizer": tokenizer,
            "encoder": ort.InferenceSession(str(encoder_path), providers=providers),
            "decoder": ort.InferenceSession(str(decoder_path), providers=providers),
            # Marian starts the decoder on the pad token and ends on </s>.
            "start_id": _token_id(directory, "decoder_start_token_id", tokenizer.pad_id),
            "eos_id": _token_id(directory, "eos_token_id", tokenizer.eos_id),
            "model_id": cfg["model_id"],
        }
        logger.info(f"[translation] loaded {cfg['model_id']} from {directory}")
        return _state
    except Exception as e:
        if not _unavailable_logged:
            logger.warning(
                f"[translation] unavailable, queries pass through untranslated: {e}")
            _unavailable_logged = True
        _state = {}
        return None


def _greedy_decode(state, text: str) -> str:
    """Runs the encoder once, then the decoder a token at a time, taking the argmax.

    transformers' generate() would do this, but it builds torch tensors. The model
    config asks for 12 beams; greedy is used instead because a search query is a
    short noun phrase where beam search buys little, and it keeps this to two ONNX
    sessions and one loop rather than a beam manager.
    """
    tokenizer, encoder, decoder = state["tokenizer"], state["encoder"], state["decoder"]

    input_ids = np.array([tokenizer.encode(text)], dtype=np.int64)
    # One sequence, no padding, so every position is attended to.
    attention_mask = np.ones_like(input_ids)

    hidden = encoder.run(["last_hidden_state"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0]

    generated = [state["start_id"]]
    for _ in range(MAX_NEW_TOKENS):
        logits = decoder.run(["logits"], {
            "input_ids": np.array([generated], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": attention_mask,
        })[0]
        step = logits[0, -1].astype(np.float64)
        # The start/pad token is a real vocabulary entry the model must never emit
        # mid-sentence; Marian's own generation config bans it the same way.
        step[state["start_id"]] = -np.inf
        next_id = int(np.argmax(step))
        if next_id == state["eos_id"]:
            break
        generated.append(next_id)

    return tokenizer.decode(generated[1:]).strip()


@functools.lru_cache(maxsize=512)
def _translate_cached(text: str) -> str:
    state = _load()
    if state is None:
        return text
    try:
        english = _greedy_decode(state, text)
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


def reset_cache() -> None:
    """Drops the loaded model and the memoised translations. For tests."""
    global _state, _unavailable_logged
    _state = None
    _unavailable_logged = False
    _translate_cached.cache_clear()
