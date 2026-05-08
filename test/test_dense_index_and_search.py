import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import torch
from omegaconf import OmegaConf

from src.index.dense import (
    DenseShardWriter,
    build_dense_faiss_index,
    load_dense_shard_manifest,
)
from src.search.dense_retrieval import DenseRetrievalHelper


class DenseIndexAndSearchTest(unittest.TestCase):
    def test_dense_shards_build_faiss_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "embed" / "all_minilm_l6_v2" / "dense_case"
            writer = DenseShardWriter(
                output_dir=output_dir,
                dim=2,
                rank=0,
                model_family="dense",
                similarity="cosine",
                normalized=False,
                shard_max_docs=2,
                value_dtype="float16",
            )
            writer.write_batch(
                ["d1", "d2", "d3"],
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            )
            writer.finalize()

            shard_infos, metadata = load_dense_shard_manifest(output_dir)
            self.assertEqual(len(shard_infos), 2)
            index, doc_ids, group_ids = build_dense_faiss_index(
                shard_infos,
                dim=int(metadata["dim"]),
                similarity=str(metadata["similarity"]),
                normalized=bool(metadata["normalized"]),
            )
            self.assertIsNone(group_ids)
            scores, doc_indexes = index.search(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32).numpy(),
                2,
            )
            self.assertEqual(doc_ids[int(doc_indexes[0, 0])], "d1")
            self.assertEqual(doc_ids[int(doc_indexes[0, 1])], "d3")
            self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))

    def test_dense_retrieval_helper_excludes_self_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_root = Path(tmp_dir) / "index" / "all_minilm_l6_v2" / "dense_eval"
            index_root.mkdir(parents=True, exist_ok=True)
            index = faiss.IndexFlatIP(2)
            index.add(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).numpy())
            faiss.write_index(index, str(index_root / "faiss.index"))
            with (index_root / "doc_ids.json").open("w", encoding="utf-8") as doc_file:
                json.dump(["q1", "d2"], doc_file)
            with (index_root / "metadata.json").open("w", encoding="utf-8") as meta_file:
                json.dump(
                    {
                        "index_kind": "dense",
                        "doc_count": 2,
                        "dim": 2,
                        "similarity": "dot",
                        "normalized": False,
                    },
                    meta_file,
                )

            cfg = OmegaConf.create(
                {
                    "model": {"name": "all_minilm_l6_v2"},
                    "encoding": {"index_dir": str(Path(tmp_dir) / "index"), "index_tag": "dense_eval"},
                    "testing": {
                        "k_list": [1],
                        "exclude_self_match": True,
                        "faiss_use_gpu": False,
                        "faiss_gpu_required": False,
                        "faiss_use_float16": False,
                        "use_cpu": True,
                        "torch_compile": False,
                        "max_windows_per_forward": None,
                    },
                }
            )
            helper = DenseRetrievalHelper(
                cfg,
                logger=__import__("logging").getLogger("dense_test"),
                index_context="evaluation",
            )
            helper.setup()
            results = helper.score_queries(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                query_ids=["q1"],
            )
            helper.shutdown()
            self.assertEqual(results[0][0], ["d2"])

    def test_dense_shards_persist_group_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "embed" / "dpr" / "grouped_case"
            writer = DenseShardWriter(
                output_dir=output_dir,
                dim=2,
                rank=0,
                model_family="dense",
                similarity="dot",
                normalized=False,
                shard_max_docs=10,
                value_dtype="float32",
            )
            writer.write_batch(
                ["p1&&&claim&&&0", "p1&&&claim&&&1", "p2&&&claim&&&0"],
                torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 1.0]]),
                doc_group_ids=["p1", "p1", "p2"],
            )
            writer.finalize()

            shard_infos, metadata = load_dense_shard_manifest(output_dir)
            index, doc_ids, group_ids = build_dense_faiss_index(
                shard_infos,
                dim=int(metadata["dim"]),
                similarity=str(metadata["similarity"]),
                normalized=bool(metadata["normalized"]),
            )
            self.assertEqual(doc_ids, ["p1&&&claim&&&0", "p1&&&claim&&&1", "p2&&&claim&&&0"])
            self.assertEqual(group_ids, ["p1", "p1", "p2"])
            scores, doc_indexes = index.search(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32).numpy(),
                3,
            )
            self.assertEqual(int(doc_indexes[0, 0]), 0)
            self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))

    def test_dense_retrieval_helper_groups_results_by_parent_doc_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_root = Path(tmp_dir) / "index" / "dpr" / "grouped_eval"
            index_root.mkdir(parents=True, exist_ok=True)
            index = faiss.IndexFlatIP(2)
            index.add(
                torch.tensor(
                    [[2.0, 0.0], [1.5, 0.0], [0.0, 1.0]],
                    dtype=torch.float32,
                ).numpy()
            )
            faiss.write_index(index, str(index_root / "faiss.index"))
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
                        "index_kind": "dense",
                        "doc_count": 3,
                        "dim": 2,
                        "similarity": "dot",
                        "normalized": False,
                        "has_group_ids": True,
                    },
                    meta_file,
                )

            cfg = OmegaConf.create(
                {
                    "model": {"name": "dpr"},
                    "encoding": {"index_dir": str(Path(tmp_dir) / "index"), "index_tag": "grouped_eval"},
                    "testing": {
                        "k_list": [2],
                        "exclude_self_match": True,
                        "faiss_use_gpu": False,
                        "faiss_gpu_required": False,
                        "faiss_use_float16": False,
                        "use_cpu": True,
                        "torch_compile": False,
                        "max_windows_per_forward": None,
                        "result_group_key": "group_id",
                        "group_candidate_pool": 3,
                    },
                }
            )
            helper = DenseRetrievalHelper(
                cfg,
                logger=__import__("logging").getLogger("dense_group_test"),
                index_context="evaluation",
            )
            helper.setup()
            results = helper.score_queries(
                torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                query_ids=["q1"],
            )
            helper.shutdown()
            self.assertEqual(results[0][0], ["d2"])

    def test_dense_retrieval_helper_allows_grouped_search_depth_below_metric_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_root = Path(tmp_dir) / "index" / "dpr" / "grouped_depth_eval"
            index_root.mkdir(parents=True, exist_ok=True)
            index = faiss.IndexFlatIP(2)
            index.add(
                torch.tensor(
                    [
                        [4.0, 0.0],
                        [3.0, 0.0],
                        [2.0, 0.0],
                        [1.0, 0.0],
                    ],
                    dtype=torch.float32,
                ).numpy()
            )
            faiss.write_index(index, str(index_root / "faiss.index"))
            with (index_root / "doc_ids.json").open("w", encoding="utf-8") as doc_file:
                json.dump(
                    ["g1&&&claim&&&0", "g1&&&claim&&&1", "g2&&&claim&&&0", "g3&&&claim&&&0"],
                    doc_file,
                )
            with (index_root / "group_ids.json").open("w", encoding="utf-8") as group_file:
                json.dump(["g1", "g1", "g2", "g3"], group_file)
            with (index_root / "metadata.json").open("w", encoding="utf-8") as meta_file:
                json.dump(
                    {
                        "index_kind": "dense",
                        "doc_count": 4,
                        "dim": 2,
                        "similarity": "dot",
                        "normalized": False,
                        "has_group_ids": True,
                    },
                    meta_file,
                )

            cfg = OmegaConf.create(
                {
                    "model": {"name": "dpr"},
                    "encoding": {
                        "index_dir": str(Path(tmp_dir) / "index"),
                        "index_tag": "grouped_depth_eval",
                    },
                    "testing": {
                        "k_list": [10],
                        "exclude_self_match": False,
                        "faiss_use_gpu": False,
                        "faiss_gpu_required": False,
                        "faiss_use_float16": False,
                        "use_cpu": True,
                        "torch_compile": False,
                        "max_windows_per_forward": None,
                        "result_group_key": "group_id",
                        "group_candidate_pool": 2,
                        "search_top_k": 2,
                    },
                }
            )
            helper = DenseRetrievalHelper(
                cfg,
                logger=__import__("logging").getLogger("dense_group_depth_test"),
                index_context="evaluation",
            )
            helper.setup()
            results = helper.score_queries(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
            helper.shutdown()
            self.assertEqual(results[0][0], ["g1"])

    def test_dense_retrieval_helper_falls_back_to_cpu_when_faiss_gpu_clone_fails(self) -> None:
        cfg = OmegaConf.create(
            {
                "testing": {
                    "k_list": [1],
                    "exclude_self_match": False,
                    "faiss_use_gpu": True,
                    "faiss_gpu_required": False,
                    "faiss_use_float16": False,
                    "use_cpu": False,
                    "torch_compile": False,
                    "max_windows_per_forward": None,
                }
            }
        )
        helper = DenseRetrievalHelper(
            cfg,
            logger=__import__("logging").getLogger("dense_gpu_fallback_test"),
            index_context="evaluation",
        )
        index = faiss.IndexFlatIP(2)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("src.search.dense_retrieval.faiss.StandardGpuResources", return_value=object()),
            patch(
                "src.search.dense_retrieval.faiss.GpuClonerOptions",
                return_value=type("GpuClonerOptionsStub", (), {"useFloat16": False})(),
            ),
            patch(
                "src.search.dense_retrieval.faiss.index_cpu_to_gpu",
                side_effect=RuntimeError("gpu oom"),
            ),
        ):
            search_index = helper._maybe_clone_index_to_gpu(index, device_index=0)
        self.assertIs(search_index, index)

    def test_dense_retrieval_helper_shards_faiss_index_across_visible_gpus(self) -> None:
        cfg = OmegaConf.create(
            {
                "testing": {
                    "k_list": [1],
                    "exclude_self_match": False,
                    "faiss_use_gpu": True,
                    "faiss_gpu_required": False,
                    "faiss_gpu_shard": True,
                    "faiss_use_float16": False,
                    "use_cpu": False,
                    "torch_compile": False,
                    "max_windows_per_forward": None,
                }
            }
        )
        helper = DenseRetrievalHelper(
            cfg,
            logger=__import__("logging").getLogger("dense_gpu_shard_test"),
            index_context="evaluation",
        )
        index = faiss.IndexFlatIP(2)
        multi_clone_options = type(
            "GpuMultipleClonerOptionsStub",
            (),
            {"useFloat16": False, "shard": False},
        )()
        resources = [object(), object(), object()]
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=3),
            patch(
                "src.search.dense_retrieval.faiss.StandardGpuResources",
                side_effect=resources,
            ),
            patch(
                "src.search.dense_retrieval.faiss.GpuMultipleClonerOptions",
                return_value=multi_clone_options,
            ),
            patch(
                "src.search.dense_retrieval.faiss.index_cpu_to_gpu_multiple_py",
                return_value="sharded-index",
            ) as multi_clone,
            patch("src.search.dense_retrieval.faiss.index_cpu_to_gpu") as single_clone,
        ):
            search_index = helper._maybe_clone_index_to_gpu(index, device_index=0)
        self.assertEqual(search_index, "sharded-index")
        self.assertTrue(bool(multi_clone_options.shard))
        self.assertFalse(bool(multi_clone_options.useFloat16))
        multi_clone.assert_called_once()
        args, kwargs = multi_clone.call_args
        self.assertEqual(args[0], resources)
        self.assertIs(args[1], index)
        self.assertEqual(kwargs["gpus"], [0, 1, 2])
        self.assertIs(kwargs["co"], multi_clone_options)
        single_clone.assert_not_called()
