"""Backward-compatible evaluate entrypoint.

Routes retrieval evaluation to ``script/evaluation.py`` and supports
``--benchmark nanobeir`` / ``--benchmark mteb`` / ``--benchmark true_mteb``
benchmark entrypoints.
"""

from __future__ import annotations

import sys

_SUPPORTED_BENCHMARK_NAMES: set[str] = {"nanobeir", "mteb", "true_mteb"}


def _extract_benchmark_argument(argv: list[str]) -> tuple[str | None, list[str]]:
    benchmark_name: str | None = None
    forwarded_argv: list[str] = [argv[0]]

    index: int = 1
    while index < len(argv):
        arg: str = argv[index]
        if arg == "--benchmark":
            if index + 1 >= len(argv):
                raise ValueError("Missing value for --benchmark.")
            benchmark_name = str(argv[index + 1]).strip().lower()
            index += 2
            continue
        if arg.startswith("--benchmark="):
            benchmark_name = str(arg.split("=", maxsplit=1)[1]).strip().lower()
            index += 1
            continue
        forwarded_argv.append(arg)
        index += 1

    if benchmark_name == "":
        raise ValueError("--benchmark must not be empty.")
    return benchmark_name, forwarded_argv


def main() -> None:
    benchmark_name: str | None
    forwarded_argv: list[str]
    benchmark_name, forwarded_argv = _extract_benchmark_argument(sys.argv)
    sys.argv = forwarded_argv

    if benchmark_name is None:
        from script.evaluation import main as retrieval_main

        retrieval_main()
        return

    if benchmark_name == "nanobeir":
        from script.evaluate_nanobeir import main as nanobeir_main

        nanobeir_main()
        return

    if benchmark_name == "mteb":
        from script.evaluate_mteb import main as mteb_main

        mteb_main()
        return

    if benchmark_name == "true_mteb":
        from script.evaluate_true_mteb import main as true_mteb_main

        true_mteb_main()
        return

    supported_values: str = ", ".join(sorted(_SUPPORTED_BENCHMARK_NAMES))
    raise ValueError(
        f"Unsupported benchmark {benchmark_name!r}. Supported values: {supported_values}."
    )


if __name__ == "__main__":
    main()
