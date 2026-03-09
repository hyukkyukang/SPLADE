import argparse
import json
from pathlib import Path

from src.prototype.embeddinggemma_lsr.artifacts import VOCAB_STATS_FILENAME, write_json
from src.prototype.embeddinggemma_lsr.vocab_audit import audit_vocab_stats, load_vocab_stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a generated EmbeddingGemma vocabulary from vocab_stats.json."
    )
    parser.add_argument(
        "--vocab-artifact-dir",
        type=str,
        default="outputs/model_creation/embeddinggemma_splade/vocab",
    )
    parser.add_argument("--vocab-stats-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    return parser


def main() -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()

    vocab_stats_path: Path
    if args.vocab_stats_path is not None and str(args.vocab_stats_path).strip():
        vocab_stats_path = Path(str(args.vocab_stats_path))
    else:
        vocab_stats_path = Path(str(args.vocab_artifact_dir)) / VOCAB_STATS_FILENAME

    vocab_stats: dict = load_vocab_stats(vocab_stats_path)
    report: dict = audit_vocab_stats(vocab_stats)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_path is not None and str(args.output_path).strip():
        write_json(Path(str(args.output_path)), report)


if __name__ == "__main__":
    main()
