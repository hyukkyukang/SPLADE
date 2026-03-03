"""Backward-compatible entrypoint for EmbeddingGemma target vocab build."""

from script.preprocess.build_embeddinggemma_lsr_vocab import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
