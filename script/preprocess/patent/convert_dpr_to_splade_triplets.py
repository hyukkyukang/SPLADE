#!/usr/bin/env python3
"""Convert DPR-format patent JSON to SPLADE inline-triplet JSONL.

Reads the coworker's DPR training data (negative1_ko_en_20251202_filtered.json),
filters to English-only examples, splits into train/val, and writes JSONL files
in the {query_id, query, positive, negative} format that SPLADE's MSMARCODataset
can consume directly.

Usage:
    python script/preprocess/patent/convert_dpr_to_splade_triplets.py \
        --input data/negative1_ko_en_20251202_filtered.json \
        --output-dir data \
        --val-size 2048 \
        --seed 42
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from random import Random

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Hangul syllables + Jamo blocks
_HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")


def contains_korean(text: str) -> bool:
    return bool(_HANGUL_RE.search(text))


def convert_example(example: dict, idx: int) -> dict | None:
    """Convert a single DPR example to SPLADE inline triplet format.

    Returns None if the example should be skipped.
    """
    question = example.get("question", "")
    if not question or contains_korean(question):
        return None

    positive_ctxs = example.get("positive_ctxs", [])
    negative_ctxs = example.get("negative_ctxs", [])

    if not positive_ctxs or not negative_ctxs:
        return None

    pos_text = positive_ctxs[0].get("text", "").strip()
    neg_text = negative_ctxs[0].get("text", "").strip()

    if not pos_text or not neg_text:
        return None

    # Skip if positive or negative contain Korean
    if contains_korean(pos_text) or contains_korean(neg_text):
        return None

    return {
        "query_id": str(idx),
        "query": question.strip(),
        "positive": pos_text,
        "negative": neg_text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="data/negative1_ko_en_20251202_filtered.json",
        help="Path to the DPR JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2048,
        help="Number of examples to hold out for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "patent_dpr_en_train.jsonl"
    val_path = output_dir / "patent_dpr_en_val.jsonl"

    # --- Load and filter ---
    logger.info("Loading %s ...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info("Loaded %d raw examples", len(raw_data))

    english_examples: list[dict] = []
    skipped_korean = 0
    skipped_missing = 0
    skipped_cross_lingual = 0

    for i, example in enumerate(raw_data):
        question = example.get("question", "")
        if not question or contains_korean(question):
            skipped_korean += 1
            continue

        positive_ctxs = example.get("positive_ctxs", [])
        negative_ctxs = example.get("negative_ctxs", [])
        if not positive_ctxs or not negative_ctxs:
            skipped_missing += 1
            continue

        pos_text = positive_ctxs[0].get("text", "").strip()
        neg_text = negative_ctxs[0].get("text", "").strip()
        if not pos_text or not neg_text:
            skipped_missing += 1
            continue

        if contains_korean(pos_text) or contains_korean(neg_text):
            skipped_cross_lingual += 1
            continue

        english_examples.append({
            "query": question.strip(),
            "positive": pos_text,
            "negative": neg_text,
        })

    logger.info(
        "Filtered to %d English examples (skipped: %d Korean, %d missing, %d cross-lingual)",
        len(english_examples),
        skipped_korean,
        skipped_missing,
        skipped_cross_lingual,
    )

    if len(english_examples) < args.val_size:
        logger.error(
            "Not enough English examples (%d) for val_size=%d",
            len(english_examples),
            args.val_size,
        )
        sys.exit(1)

    # --- Shuffle and split ---
    rng = Random(args.seed)
    rng.shuffle(english_examples)

    val_examples = english_examples[: args.val_size]
    train_examples = english_examples[args.val_size :]

    logger.info("Train: %d examples, Val: %d examples", len(train_examples), len(val_examples))

    # --- Write JSONL ---
    def write_jsonl(path: Path, examples: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for idx, ex in enumerate(examples):
                row = {
                    "query_id": str(idx),
                    "query": ex["query"],
                    "positive": ex["positive"],
                    "negative": ex["negative"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d examples to %s (%.1f MB)", len(examples), path, path.stat().st_size / 1e6)

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    logger.info("Done.")


if __name__ == "__main__":
    main()
