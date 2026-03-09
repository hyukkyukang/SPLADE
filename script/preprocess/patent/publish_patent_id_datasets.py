import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


KOR_TEST_FILES: tuple[str, ...] = (
    "new_kor_Electrical_Electronics_testset_1_list.json",
    "new_kor_Electrical_Electronics_testset_2_list.json",
    "new_kor_Electrical_Electronics_testset_3_list.json",
)

US_TEST_FILES: tuple[str, ...] = (
    "new_us_Electrical_Electronics_testset_1_list.json",
    "new_us_Electrical_Electronics_testset_2_list.json",
    "new_us_Electrical_Electronics_testset_3_list.json",
)

US_TRAIN_FILE: str = "usc102103_train.json"


def _sha1_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array at {path.as_posix()}")
    return data


def _build_test_rows(data_dir: Path, file_names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in file_names:
        path = data_dir / file_name
        for row in _load_json_array(path):
            question_id = str(row["question_id"])
            label_id = [str(doc_id) for doc_id in row["label_id"]]
            rows.append({"question_id": question_id, "label_id": label_id})
    return rows


def _build_us_train_rows(data_dir: Path) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    path = data_dir / US_TRAIN_FILE
    for row in _load_json_array(path):
        question_text = str(row.get("question", ""))
        question_id = _sha1_id("q_", question_text)

        label_ids: list[str] = []
        for ctx in row.get("positive_ctxs", []):
            if not isinstance(ctx, dict):
                continue
            doc_text = str(ctx.get("text", ""))
            label_ids.append(_sha1_id("d_", doc_text))

        train_rows.append(
            {"question_id": question_id, "label_id": _dedupe_preserve_order(label_ids)}
        )
    return train_rows


def _build_us_train_rows_from_parquet(train_parquet_path: Path) -> list[dict[str, Any]]:
    dataset = load_dataset(
        "parquet",
        data_files={"train": train_parquet_path.as_posix()},
        split="train",
    )
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(
            {
                "question_id": str(row["question_id"]),
                "label_id": [str(doc_id) for doc_id in row.get("label_id", [])],
            }
        )
    return rows


def _load_split_rows_from_hub(repo_id: str, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset(repo_id, split=split)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(
            {
                "question_id": str(row["question_id"]),
                "label_id": [str(doc_id) for doc_id in row.get("label_id", [])],
            }
        )
    return rows


def _build_kr_dataset(data_dir: Path) -> DatasetDict:
    test_rows = _build_test_rows(data_dir=data_dir, file_names=KOR_TEST_FILES)
    return DatasetDict({"test": Dataset.from_list(test_rows)})


def _build_us_dataset(
    data_dir: Path,
    *,
    train_parquet_path: Path | None = None,
    test_source_repo: str | None = None,
    test_source_split: str = "test",
) -> DatasetDict:
    if train_parquet_path is None:
        train_rows = _build_us_train_rows(data_dir=data_dir)
    else:
        train_rows = _build_us_train_rows_from_parquet(train_parquet_path=train_parquet_path)
    if test_source_repo is None:
        test_rows = _build_test_rows(data_dir=data_dir, file_names=US_TEST_FILES)
    else:
        test_rows = _load_split_rows_from_hub(repo_id=test_source_repo, split=test_source_split)
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )


def _validate_schema(dataset_dict: DatasetDict) -> None:
    for split_name, split_dataset in dataset_dict.items():
        if split_dataset.column_names != ["question_id", "label_id"]:
            raise ValueError(
                f"Unexpected schema in split={split_name}: {split_dataset.column_names}"
            )


def _print_dataset_stats(name: str, dataset_dict: DatasetDict) -> None:
    print(f"[{name}]")
    for split_name, split_dataset in dataset_dict.items():
        print(f"  - {split_name}: {len(split_dataset)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and optionally push patent ID-only datasets to Hugging Face Hub."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/patent"),
        help="Directory containing patent JSON files.",
    )
    parser.add_argument(
        "--kr-repo-id",
        type=str,
        default="Hyukkyu/patent-kr",
        help="Target dataset repo for Korean test split.",
    )
    parser.add_argument(
        "--us-repo-id",
        type=str,
        default="Hyukkyu/patent-us",
        help="Target dataset repo for US train/test splits.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build datasets locally but do not push to Hub.",
    )
    parser.add_argument(
        "--us-train-parquet",
        type=Path,
        default=None,
        help="Optional parquet file with train question_id/label_id rows to use instead of hashing usc102103_train.json.",
    )
    parser.add_argument(
        "--us-test-source-repo",
        type=str,
        default=None,
        help="Optional HF dataset repo to copy the US test split from instead of local JSON files.",
    )
    parser.add_argument(
        "--us-test-source-split",
        type=str,
        default="test",
        help="Split name used with --us-test-source-repo.",
    )
    parser.add_argument(
        "--skip-kr",
        action="store_true",
        help="Skip building or uploading the KR dataset.",
    )
    args = parser.parse_args()

    kr_dataset = None if args.skip_kr else _build_kr_dataset(data_dir=args.data_dir)
    us_dataset = _build_us_dataset(
        data_dir=args.data_dir,
        train_parquet_path=args.us_train_parquet,
        test_source_repo=args.us_test_source_repo,
        test_source_split=args.us_test_source_split,
    )

    if kr_dataset is not None:
        _validate_schema(kr_dataset)
    _validate_schema(us_dataset)
    if kr_dataset is not None:
        _print_dataset_stats("Hyukkyu/patent-kr", kr_dataset)
    _print_dataset_stats("Hyukkyu/patent-us", us_dataset)

    if args.skip_upload:
        print("Skipping upload (--skip-upload).")
        return

    if not args.token:
        raise ValueError("Missing token. Set HF_TOKEN or pass --token.")

    HfApi(token=args.token).whoami()
    if kr_dataset is not None:
        kr_dataset.push_to_hub(args.kr_repo_id, token=args.token)
    us_dataset.push_to_hub(args.us_repo_id, token=args.token)
    print("Upload completed.")


if __name__ == "__main__":
    main()
