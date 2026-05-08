#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any

import ijson


@dataclass
class OutputSpec:
    top_k: int
    output_path: Path
    temp_path: Path
    handle: object
    first_entry: bool = True


@dataclass(frozen=True, slots=True)
class FlattenedTermEntry:
    order: int
    term: str
    weight: float
    source_key: str | None = None


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def resolve_term_payload_mode(term_weights: Any) -> str:
    if not isinstance(term_weights, dict):
        raise ValueError(f"Expected a dict payload, got: {type(term_weights).__name__}")
    if not term_weights:
        return "empty"

    saw_flat_terms: bool = False
    saw_source_terms: bool = False
    value: Any
    for value in term_weights.values():
        if isinstance(value, dict):
            saw_source_terms = True
            nested_weight: Any
            for nested_weight in value.values():
                if not _is_numeric_value(nested_weight):
                    raise ValueError("Nested source-token payload values must be numeric weights.")
            continue
        if _is_numeric_value(value):
            saw_flat_terms = True
            continue
        raise ValueError(f"Unsupported term payload value type: {type(value).__name__}")

    if saw_flat_terms and saw_source_terms:
        raise ValueError("Mixed flat and source-token term payloads are not supported.")
    if saw_source_terms:
        return "source_token_terms"
    return "flat_terms"


def flatten_term_payload(term_weights: Any) -> tuple[str, list[FlattenedTermEntry]]:
    payload_mode: str = resolve_term_payload_mode(term_weights)
    if payload_mode == "empty":
        return payload_mode, []

    entries: list[FlattenedTermEntry] = []
    order: int = 0
    if payload_mode == "flat_terms":
        term: Any
        weight: Any
        for term, weight in term_weights.items():
            entries.append(
                FlattenedTermEntry(
                    order=order,
                    term=str(term),
                    weight=float(weight),
                )
            )
            order += 1
        return payload_mode, entries

    source_key: Any
    source_terms: Any
    for source_key, source_terms in term_weights.items():
        if not isinstance(source_terms, dict):
            raise ValueError("Expected nested dict payload for source-token terms.")
        term: Any
        weight: Any
        for term, weight in source_terms.items():
            entries.append(
                FlattenedTermEntry(
                    order=order,
                    source_key=str(source_key),
                    term=str(term),
                    weight=float(weight),
                )
            )
            order += 1
    return payload_mode, entries


def select_top_term_entries(
    entries: list[FlattenedTermEntry],
    *,
    top_k: int,
) -> list[FlattenedTermEntry]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    sorted_entries = sorted(entries, key=lambda entry: (-entry.weight, entry.order))
    return sorted_entries[:top_k]


def rebuild_term_payload(
    entries: list[FlattenedTermEntry],
    *,
    payload_mode: str,
) -> dict[str, Any]:
    if payload_mode in {"flat_terms", "empty"}:
        result: dict[str, float] = {}
        entry: FlattenedTermEntry
        for entry in entries:
            result[entry.term] = float(entry.weight)
        return result
    if payload_mode != "source_token_terms":
        raise ValueError(f"Unsupported payload_mode: {payload_mode!r}")
    result_nested: dict[str, dict[str, float]] = {}
    entry: FlattenedTermEntry
    for entry in entries:
        if entry.source_key is None:
            raise ValueError("source_token_terms entries must include a source_key")
        bucket = result_nested.setdefault(entry.source_key, {})
        bucket[entry.term] = float(entry.weight)
    return result_nested


def truncate_term_payload(term_weights: Any, *, top_k: int) -> dict[str, Any]:
    payload_mode, entries = flatten_term_payload(term_weights)
    selected_entries = select_top_term_entries(entries, top_k=top_k)
    return rebuild_term_payload(selected_entries, payload_mode=payload_mode)


def parse_top_k_output(value: str) -> tuple[int, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "--top-k-output must be formatted as <top_k>:<output_path>"
        )
    top_k_text, output_path_text = value.split(":", 1)
    try:
        top_k = int(top_k_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid top_k value: {top_k_text}") from exc
    if top_k <= 0:
        raise argparse.ArgumentTypeError("top_k must be positive")
    output_path = Path(output_path_text)
    return top_k, output_path


def open_outputs(spec_pairs: list[tuple[int, Path]]) -> list[OutputSpec]:
    outputs: list[OutputSpec] = []
    for top_k, output_path in sorted(spec_pairs, key=lambda item: item[0]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(output_path.name + ".tmp")
        handle = temp_path.open("w", encoding="utf-8")
        handle.write("{")
        outputs.append(
            OutputSpec(
                top_k=top_k,
                output_path=output_path,
                temp_path=temp_path,
                handle=handle,
            )
        )
    return outputs


def write_entry(spec: OutputSpec, doc_id: str, term_weights: dict[str, Any]) -> None:
    if spec.first_entry:
        spec.handle.write("\n")
        spec.first_entry = False
    else:
        spec.handle.write(",\n")
    json.dump(doc_id, spec.handle, ensure_ascii=False)
    spec.handle.write(": ")
    json.dump(term_weights, spec.handle, ensure_ascii=False)


def finalize_outputs(outputs: list[OutputSpec]) -> None:
    for spec in outputs:
        if not spec.first_entry:
            spec.handle.write("\n")
        spec.handle.write("}\n")
        spec.handle.close()
        spec.temp_path.replace(spec.output_path)


def cleanup_outputs(outputs: list[OutputSpec]) -> None:
    for spec in outputs:
        try:
            spec.handle.close()
        except Exception:
            pass
        try:
            spec.temp_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create top-k patent term JSON exports from an untruncated merged export."
    )
    parser.add_argument("--input", required=True, help="Path to the untruncated merged JSON export.")
    parser.add_argument(
        "--top-k-output",
        required=True,
        nargs="+",
        type=parse_top_k_output,
        help="One or more <top_k>:<output_path> pairs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Log progress every N documents. Set to 0 to disable.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_specs = open_outputs(list(args.top_k_output))
    max_top_k = max(spec.top_k for spec in output_specs)
    document_count = 0
    payload_mode_counts: Counter[str] = Counter()

    try:
        with input_path.open("rb") as handle:
            for doc_id, term_weights in ijson.kvitems(handle, "", use_float=True):
                document_count += 1
                payload_mode, flattened_entries = flatten_term_payload(term_weights)
                payload_mode_counts[payload_mode] += 1
                top_entries = select_top_term_entries(flattened_entries, top_k=max_top_k)
                for spec in output_specs:
                    write_entry(
                        spec,
                        str(doc_id),
                        rebuild_term_payload(top_entries[: spec.top_k], payload_mode=payload_mode),
                    )
                if args.progress_every > 0 and document_count % int(args.progress_every) == 0:
                    print(f"processed {document_count} documents")
        finalize_outputs(output_specs)
    except Exception:
        cleanup_outputs(output_specs)
        raise

    print(json.dumps({
        "document_count": document_count,
        "payload_mode_counts": {
            str(mode): int(count) for mode, count in sorted(payload_mode_counts.items())
        },
        "outputs": [
            {"top_k": spec.top_k, "path": str(spec.output_path)} for spec in output_specs
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
