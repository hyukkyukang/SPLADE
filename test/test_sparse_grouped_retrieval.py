import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.search.retrieval import IndexedRetrievalHelper


class SparseGroupedRetrievalTest(unittest.TestCase):
    def _write_grouped_sparse_index(self, index_root: Path) -> None:
        index_root.mkdir(parents=True, exist_ok=True)
        np.save(index_root / "term_ptr.npy", np.array([0, 3], dtype=np.int64))
        np.save(index_root / "post_doc_ids.npy", np.array([0, 1, 2], dtype=np.int32))
        np.save(
            index_root / "post_weights.npy",
            np.array([4.0, 3.0, 2.0], dtype=np.float32),
        )
        np.save(index_root / "term_max.npy", np.array([4.0], dtype=np.float32))
        np.save(index_root / "block_max.npy", np.array([4.0], dtype=np.float32))
        np.save(index_root / "block_ptr.npy", np.array([0, 1], dtype=np.int64))
        with (index_root / "doc_ids.json").open("w", encoding="utf-8") as doc_file:
            json.dump(
                ["q1&&&claim&&&0", "q1&&&claim&&&1", "d2&&&claim&&&0"],
                doc_file,
            )
        with (index_root / "group_ids.json").open("w", encoding="utf-8") as group_file:
            json.dump(["q1", "q1", "d2"], group_file)
        with (index_root / "metadata.json").open("w", encoding="utf-8") as meta_file:
            json.dump(
                {
                    "vocab_size": 1,
                    "doc_count": 3,
                    "nnz": 3,
                    "value_dtype": "float32",
                    "encoded_value_dtype": "float32",
                    "block_size": 4,
                    "has_block_max": True,
                    "has_group_ids": True,
                },
                meta_file,
            )

    def _build_cfg(
        self,
        root: Path,
        *,
        search_top_k: int | None = None,
        group_candidate_pool: int = 3,
    ):
        testing_cfg = {
            "k_list": [2],
            "exclude_self_match": True,
            "gpu_sparsify": False,
            "scoring_workers": 0,
            "use_cpu": True,
            "scoring_method": "full",
            "scoring_backend": "threads",
            "query_exclude_token_ids": [],
            "sparse_min_weight": 0.0,
            "sparse_top_k": None,
            "wand_block_size": 4,
            "torch_compile": False,
            "max_windows_per_forward": None,
            "result_group_key": "group_id",
            "group_candidate_pool": int(group_candidate_pool),
        }
        if search_top_k is not None:
            testing_cfg["search_top_k"] = int(search_top_k)
        return OmegaConf.create(
            {
                "model": {"name": "splade_test"},
                "encoding": {
                    "index_dir": str(root / "index"),
                    "index_tag": "grouped_eval",
                },
                "testing": testing_cfg,
            }
        )

    def test_sparse_retrieval_helper_groups_results_by_parent_doc_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_root = Path(tmp_dir) / "index" / "splade_test" / "grouped_eval"
            self._write_grouped_sparse_index(index_root)
            helper = IndexedRetrievalHelper(
                self._build_cfg(Path(tmp_dir)),
                logger=__import__("logging").getLogger("sparse_group_test"),
                index_context="evaluation",
            )
            helper.setup()
            results = helper.score_queries(
                torch.tensor([[1.0]], dtype=torch.float32),
                query_ids=["q1"],
            )
            helper.shutdown()
            self.assertEqual(results[0][0], ["d2"])

    def test_sparse_retrieval_grouping_respects_search_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_root = Path(tmp_dir) / "index" / "splade_test" / "grouped_eval"
            self._write_grouped_sparse_index(index_root)
            helper = IndexedRetrievalHelper(
                self._build_cfg(
                    Path(tmp_dir),
                    search_top_k=2,
                    group_candidate_pool=2,
                ),
                logger=__import__("logging").getLogger("sparse_group_depth_test"),
                index_context="evaluation",
            )
            helper.setup()
            results = helper.score_queries(
                torch.tensor([[1.0]], dtype=torch.float32),
                query_ids=["q1"],
            )
            helper.shutdown()
            self.assertEqual(results[0][0], [])


if __name__ == "__main__":
    unittest.main()
