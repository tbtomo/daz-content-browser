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

Usage:
    python export_translation_model.py [--force] [--out-dir PATH] [--model MODEL_ID]

Options:
    --force         Re-export even if the model is already there
    --out-dir PATH  Parent directory for the export (default: models/)
    --model ID      HuggingFace model ID (default: staka/fugumt-ja-en)
    --keep-past     Keep decoder_with_past_model.onnx instead of deleting it

Prereqs: pip install -e ".[local_llm]"  (plus sentencepiece, for the Marian tokenizer)
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


def main():
    parser = argparse.ArgumentParser(
        description="Export the ONNX query translation model")
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if the model already exists")
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR),
                        help=f"Parent directory for the export (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, metavar="MODEL_ID",
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--keep-past", action="store_true",
                        help="Keep decoder_with_past_model.onnx (unused; ~220 MB)")
    args = parser.parse_args()

    export(Path(args.out_dir), args.force, args.model, args.keep_past)


if __name__ == "__main__":
    main()
