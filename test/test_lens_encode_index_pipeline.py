import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def _install_fake_numba() -> None:
    try:
        import numba  # noqa: F401
    except ImportError:
        def _njit(func=None, *args, **kwargs):  # type: ignore[no-untyped-def]
            if func is not None and callable(func):
                return func

            def decorator(inner):  # type: ignore[no-untyped-def]
                return inner

            return decorator

        sys.modules["numba"] = types.SimpleNamespace(njit=_njit)


def _install_fake_hydra() -> None:
    try:
        import hydra  # noqa: F401
    except ImportError:
        def _main(*args, **kwargs):  # type: ignore[no-untyped-def]
            def decorator(func):  # type: ignore[no-untyped-def]
                @wraps(func)
                def wrapped(*func_args, **func_kwargs):
                    return func(*func_args, **func_kwargs)

                return wrapped

            return decorator

        sys.modules["hydra"] = types.SimpleNamespace(main=_main)


def _install_fake_torchmetrics() -> None:
    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        class _FakeMetricCollection(dict):
            def __init__(self, metrics=None, prefix=""):  # type: ignore[no-untyped-def]
                super().__init__(metrics or {})
                self.prefix = prefix

            def register_buffer(  # type: ignore[no-untyped-def]
                self, name, value, persistent=False
            ):
                setattr(self, name, value)

            def update(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return None

            def reset(self):  # type: ignore[no-untyped-def]
                return None

        class _FakeMetric:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return None

            def update(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return None

        retrieval_module = types.SimpleNamespace(
            RetrievalHitRate=_FakeMetric,
            RetrievalMAP=_FakeMetric,
            RetrievalMRR=_FakeMetric,
            RetrievalNormalizedDCG=_FakeMetric,
            RetrievalRecall=_FakeMetric,
        )
        torchmetrics_module = types.SimpleNamespace(
            MetricCollection=_FakeMetricCollection,
        )
        sys.modules["torchmetrics"] = torchmetrics_module
        sys.modules["torchmetrics.retrieval"] = retrieval_module


def _load_index_script_module():
    module_name = "splade_script_index_pipeline_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).resolve().parents[1] / "script" / "index.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_install_fake_numba()
_install_fake_hydra()
_install_fake_torchmetrics()

from src.index.sparse import SparseShardWriter

IndexedRetrievalHelper = importlib.import_module(
    "src.search.retrieval"
).IndexedRetrievalHelper


class _StubEncoder:
    def __init__(self) -> None:
        self.vocab_size = 8
        self.mlm = types.SimpleNamespace(
            config=types.SimpleNamespace(pad_token_id=0)
        )

    def resolve_output_exclude_ids(
        self, exclude_token_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if exclude_token_ids is None or int(exclude_token_ids.numel()) == 0:
            return torch.empty((0,), dtype=torch.long)
        mapping = {26: 1, 27: 3, 28: 7}
        resolved = sorted(
            {
                mapping[int(token_id)]
                for token_id in exclude_token_ids.flatten().tolist()
                if int(token_id) in mapping
            }
        )
        return torch.tensor(resolved, dtype=torch.long)


class _StubClusteredModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _StubEncoder()
        self.query_pooling = "max"
        self.register_parameter(
            "_dummy_parameter",
            torch.nn.Parameter(torch.zeros((), dtype=torch.float32)),
        )

    def encode_queries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask, pooling_mask
        batch_size = int(input_ids.shape[0])
        reps = torch.zeros((batch_size, self.encoder.vocab_size), dtype=torch.float32)
        for row_idx, row in enumerate(input_ids.tolist()):
            if 26 in row:
                reps[row_idx, 1] = 5.0
            if 5 in row:
                reps[row_idx, 5] = 2.0
        return reps


class LensEncodeIndexPipelineTest(unittest.TestCase):
    def _write_clustered_encode_shard(self, encode_path: Path) -> None:
        writer = SparseShardWriter(
            output_dir=encode_path,
            vocab_size=8,
            rank=0,
            top_k=16,
            min_weight=0.0,
            exclude_output_ids=[1],
            source_exclude_token_ids=[26],
            model_family="lens",
            compact_head_alignment="latent_cluster",
            output_token_aligned=False,
            shard_max_docs=32,
            value_dtype="float32",
        )
        writer.write_sparse_csr_batch(
            ["doc-a", "doc-b"],
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 5], dtype=np.int32),
            np.array([5.0, 4.0], dtype=np.float32),
        )
        writer.finalize()

    def test_clustered_query_exclusions_affect_index_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_name = "lens_clustered"
            encode_root = root / "encode"
            encode_path = encode_root / model_name
            self._write_clustered_encode_shard(encode_path)

            index_module = _load_index_script_module()
            index_cfg = OmegaConf.create(
                {
                    "log_dir": str(root / "logs"),
                    "tag": None,
                    "model": {"name": model_name},
                    "encoding": {
                        "encode_dir": str(encode_root),
                        "index_dir": str(root / "index"),
                        "value_dtype": "float32",
                        "wand_block_size": 4,
                    },
                }
            )
            index_module.main.__wrapped__(index_cfg)

            retrieval_cfg = OmegaConf.create(
                {
                    "model": {"name": model_name},
                    "encoding": {
                        "index_dir": str(root / "index"),
                        "index_tag": None,
                    },
                    "testing": {
                        "k_list": [1],
                        "exclude_self_match": False,
                        "gpu_sparsify": False,
                        "scoring_workers": 0,
                        "use_cpu": True,
                        "scoring_method": "full",
                        "scoring_backend": "threads",
                        "query_exclude_token_ids": [26],
                        "sparse_min_weight": 0.0,
                        "sparse_top_k": None,
                        "wand_block_size": 4,
                        "torch_compile": False,
                        "max_windows_per_forward": None,
                    },
                }
            )

            helper = IndexedRetrievalHelper(
                retrieval_cfg,
                logger=types.SimpleNamespace(
                    warning=lambda *args, **kwargs: None,
                    info=lambda *args, **kwargs: None,
                ),
            )
            helper.setup()
            try:
                model = _StubClusteredModel()
                query_reps = helper.encode_queries(
                    model,
                    torch.tensor([[26, 5, 0]], dtype=torch.long),
                    torch.tensor([[1, 1, 0]], dtype=torch.long),
                    mark_step=None,
                )
                results = helper.score_queries(query_reps)

                self.assertEqual(helper._query_exclude_output_ids, [1])
                self.assertEqual(results[0][0], ["doc-b"])
            finally:
                helper.shutdown()


if __name__ == "__main__":
    unittest.main()
