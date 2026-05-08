import unittest

from datasets import Dataset

from script.preprocess.patent.build_exclusive_hard_negative_validation import (
    build_blocker_sets,
    build_exclusive_groups,
    filter_exclusive_rows,
)


def build_row(
    row_id: str,
    *,
    source_document_id: str,
    positive_document_id: str,
    query_chunk_id: str,
    positive_node_id: str,
    hard_negative_document_ids: list[str] | None = None,
    hard_negative_node_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "query_id": row_id,
        "query_text": f"query {row_id}",
        "source_document_id": source_document_id,
        "positive_document_id": positive_document_id,
        "query_chunk_id": query_chunk_id,
        "positive_node_id": positive_node_id,
        "positive_text": f"positive {row_id}",
        "hard_negative_document_ids": hard_negative_document_ids or [],
        "hard_negative_node_ids": hard_negative_node_ids or [],
        "hard_negative_texts": [],
        "hard_negative_ranks": [],
    }


class BuildExclusiveHardNegativeValidationTest(unittest.TestCase):
    def test_filters_role_agnostic_document_and_node_overlap(self) -> None:
        train_dataset = Dataset.from_list(
            [
                build_row(
                    "train",
                    source_document_id="doc-source-train",
                    positive_document_id="doc-positive-train",
                    query_chunk_id="node-query-train",
                    positive_node_id="node-positive-train",
                )
            ]
        )
        validation_dataset = Dataset.from_list(
            [
                build_row(
                    "keep",
                    source_document_id="doc-source-new",
                    positive_document_id="doc-positive-new",
                    query_chunk_id="node-query-new",
                    positive_node_id="node-positive-new",
                ),
                build_row(
                    "drop-same-doc",
                    source_document_id="doc-source-train",
                    positive_document_id="doc-other",
                    query_chunk_id="node-other-a",
                    positive_node_id="node-other-b",
                ),
                build_row(
                    "drop-cross-doc",
                    source_document_id="doc-positive-train",
                    positive_document_id="doc-other",
                    query_chunk_id="node-other-c",
                    positive_node_id="node-other-d",
                ),
                build_row(
                    "drop-cross-node",
                    source_document_id="doc-other-a",
                    positive_document_id="doc-other-b",
                    query_chunk_id="node-other-e",
                    positive_node_id="node-query-train",
                ),
            ]
        )
        groups = build_exclusive_groups()
        blockers = build_blocker_sets(train_dataset, groups)

        result = filter_exclusive_rows(validation_dataset, groups, blockers)

        self.assertEqual([row["query_id"] for row in result.rows], ["keep"])
        self.assertEqual(result.dropped_rows, 3)
        self.assertEqual(result.group_conflict_counts["document_id"], 2)
        self.assertEqual(result.group_conflict_counts["node_id"], 1)

    def test_can_block_against_train_hard_negative_ids(self) -> None:
        train_dataset = Dataset.from_list(
            [
                build_row(
                    "train",
                    source_document_id="doc-source-train",
                    positive_document_id="doc-positive-train",
                    query_chunk_id="node-query-train",
                    positive_node_id="node-positive-train",
                    hard_negative_document_ids=["doc-hard-negative-train"],
                    hard_negative_node_ids=["node-hard-negative-train"],
                )
            ]
        )
        validation_dataset = Dataset.from_list(
            [
                build_row(
                    "hard-negative-overlap",
                    source_document_id="doc-source-new",
                    positive_document_id="doc-hard-negative-train",
                    query_chunk_id="node-query-new",
                    positive_node_id="node-hard-negative-train",
                )
            ]
        )

        default_groups = build_exclusive_groups()
        default_result = filter_exclusive_rows(
            validation_dataset,
            default_groups,
            build_blocker_sets(train_dataset, default_groups),
        )
        strict_groups = build_exclusive_groups(include_train_hard_negatives=True)
        strict_result = filter_exclusive_rows(
            validation_dataset,
            strict_groups,
            build_blocker_sets(train_dataset, strict_groups),
        )

        self.assertEqual(len(default_result.rows), 1)
        self.assertEqual(len(strict_result.rows), 0)
        self.assertEqual(strict_result.dropped_rows, 1)


if __name__ == "__main__":
    unittest.main()
