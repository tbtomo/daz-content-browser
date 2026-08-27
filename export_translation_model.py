#!/usr/bin/env python
"""Export the Japanese-to-English query translator to ONNX.

Search queries are embedded against an English index, so a Japanese query is
translated before it reaches the embedding model (see src/query_translation.py).
This exports the translation model the same way export_model.py exports the
embedding model, and for the same reason: vab.spec excludes torch from the
packaged build, so inference has to run on onnxruntime.

Only the encoder and the plain decoder are kept. optimum also emits
decoder_with_past_model.onnx for cached generation, which the greedy loop in
query_translation.py does not use — dropping it saves ~220 MB in a directory that
ships with the application.

There are two backends. The default exports staka/fugumt-ja-en to ONNX. The
ctranslate2 backend downloads a model that is already converted — used for
sugoi-v4, whose training covers vocabulary the general-purpose ja-en corpora
filter out (see src/query_translation.py). That one is a plain download.

Usage:
    python export_translation_model.py                        # onnx, ~360 MB
    python export_translation_model.py --backend ctranslate2  # sugoi, ~1.1 GB

Options:
    --backend NAME  'onnx' (default) or 'ctranslate2'
    --force         Re-fetch even if the model is already there
    --out-dir PATH  Parent directory for the export (default: models/)
    --model ID      HuggingFace model ID (default: per backend)
    --keep-past     onnx only: keep decoder_with_past_model.onnx instead of deleting

Prereqs: pip install -e ".[local_llm]"  (plus sentencepiece, for the Marian tokenizer)

Note: sugoi-v4 carries NTT's JParaCrawl licence — research use, no commercial use.
It is never bundled with the application (vab.spec excludes models/), so nothing is
redistributed; it is fetched here only if you ask for it.
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_MODEL_ID = "staka/fugumt-ja-en"
DEFAULT_CT2_MODEL_ID = "entai2965/sugoi-v4-ja-en-ctranslate2"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def export(out_dir: Path, force: bool, model_id: str, keep_past: bool = False):
    model_dir = out_dir / model_id.split("/")[-1]
    encoder_path = model_dir / "encoder_model.onnx"

    if encoder_path.exists() and not force:
        log.info(f"Translation model already exported at {model_dir} "
                 f"({dir_size_mb(model_dir):.0f} MB)")
        log.info("Pass --force to re-export.")
        return

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit(
            "ERROR: optimum and transformers are required.\n"
            "  Run: pip install -e '.[local_llm]'"
        )

    if force and model_dir.exists():
        import shutil
        log.info(f"Removing existing model at {model_dir}")
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting {model_id} → {model_dir}")
    log.info("This downloads a few hundred MB from HuggingFace on first run.")

    model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
    model.save_pretrained(str(model_dir))

    # The Marian tokenizer is sentencepiece-backed and has no fast variant, so the
    # .spm files travel with the model and sentencepiece is needed at runtime too.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(model_dir))

    if not keep_past:
        past = model_dir / "decoder_with_past_model.onnx"
        if past.exists():
            log.info(f"Removing {past.name} — the greedy decoder does not use it "
                     f"({past.stat().st_size / 1_048_576:.0f} MB)")
            past.unlink()

    log.info(f"Done. Exported to {model_dir} ({dir_size_mb(model_dir):.0f} MB)")
    log.info("Japanese queries will now be translated before they are embedded.")


def download_ctranslate2(out_dir: Path, force: bool, model_id: str):
    """Fetches an already-converted CTranslate2 model. There is no export step."""
    model_dir = out_dir / model_id.split("/")[-1]
    if (model_dir / "model.bin").exists() and not force:
        log.info(f"Translation model already present at {model_dir} "
                 f"({dir_size_mb(model_dir):.0f} MB)")
        log.info("Pass --force to re-download.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub is required.\n"
                 "  Run: pip install huggingface-hub")

    log.info(f"Downloading {model_id} -> {model_dir}")
    log.info("sugoi-v4 is ~1.1 GB and licensed for research use only (NTT/JParaCrawl).")
    snapshot_download(repo_id=model_id, local_dir=str(model_dir))
    log.info(f"Done. {model_dir} ({dir_size_mb(model_dir):.0f} MB)")
    log.info("Set QUERY_TRANSLATION_BACKEND=ctranslate2 in .env to use it.")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the query translation model")
    parser.add_argument("--backend", default="onnx", choices=("onnx", "ctranslate2"),
                        help="Which translation backend to fetch (default: onnx)")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if the model already exists")
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR),
                        help=f"Parent directory for the export (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--model", default=None, metavar="MODEL_ID",
                        help="HuggingFace model ID (default: per backend)")
    parser.add_argument("--keep-past", action="store_true",
                        help="onnx only: keep decoder_with_past_model.onnx (~220 MB)")
    args = parser.parse_args()

    if args.backend == "ctranslate2":
        download_ctranslate2(Path(args.out_dir), args.force,
                             args.model or DEFAULT_CT2_MODEL_ID)
    else:
        export(Path(args.out_dir), args.force,
               args.model or DEFAULT_MODEL_ID, args.keep_past)


if __name__ == "__main__":
    main()
