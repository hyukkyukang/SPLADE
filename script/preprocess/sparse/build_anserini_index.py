#!/usr/bin/env python3
"""Build an Anserini impact index from exported sparse JSONL files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path


DEFAULT_ANSERINI_VERSION = "1.6.0"
DEFAULT_ANSERINI_URL = (
    "https://repo1.maven.org/maven2/io/anserini/anserini/"
    f"{DEFAULT_ANSERINI_VERSION}/anserini-{DEFAULT_ANSERINI_VERSION}-fatjar.jar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of document JSONL files exported for JsonVectorCollection.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Output Lucene/Anserini index directory.",
    )
    parser.add_argument(
        "--jar-path",
        type=Path,
        default=Path("tools/anserini/anserini-1.6.0-fatjar.jar"),
        help="Path to Anserini fatjar. Downloaded automatically if missing.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Indexing thread count.",
    )
    parser.add_argument(
        "--memory-buffer-mb",
        type=int,
        default=4096,
        help="Lucene indexing memory buffer in MB.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Lucene optimize at the end of indexing.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append documents into an existing index instead of creating a new one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing target index directory first.",
    )
    return parser.parse_args()


def ensure_jar(jar_path: Path) -> None:
    if jar_path.exists():
        return
    jar_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DEFAULT_ANSERINI_URL} -> {jar_path}")
    curl_path = shutil.which("curl")
    if curl_path is not None:
        subprocess.run(
            [curl_path, "-L", "--fail", "-o", str(jar_path), DEFAULT_ANSERINI_URL],
            check=True,
        )
        return
    wget_path = shutil.which("wget")
    if wget_path is not None:
        subprocess.run(
            [wget_path, "-O", str(jar_path), DEFAULT_ANSERINI_URL],
            check=True,
        )
        return
    urllib.request.urlretrieve(DEFAULT_ANSERINI_URL, jar_path)


def main() -> None:
    args = parse_args()
    ensure_jar(args.jar_path)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")
    if args.index_dir.exists():
        if args.append:
            pass
        elif not args.overwrite:
            raise FileExistsError(
                f"Index dir already exists: {args.index_dir}. Use --overwrite to replace it."
            )
        else:
            subprocess.run(["rm", "-rf", str(args.index_dir)], check=True)
    args.index_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java",
        "-cp",
        str(args.jar_path),
        "io.anserini.index.IndexCollection",
        "-collection",
        "JsonVectorCollection",
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        str(int(args.threads)),
        "-input",
        str(args.input_dir),
        "-index",
        str(args.index_dir),
        "-impact",
        "-pretokenized",
        "-memoryBuffer",
        str(int(args.memory_buffer_mb)),
    ]
    if args.optimize:
        cmd.append("-optimize")
    if args.append:
        cmd.append("-append")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
