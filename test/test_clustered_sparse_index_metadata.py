import importlib.util
import json
import sys
import tempfile
import types
import unittest
from functools import wraps
from pathlib import Path

import numpy as np
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


def _load_index_script_module():
    module_name = "splade_script_index_test"
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


_install_fake_hydra()

from src.index.sparse import SparseShardWriter, load_shard_manifest
from src.search.index import load_inverted_index


class ClusteredSparseIndexMetadataTest(unittest.TestCase):
    def _write_clustered_encode_shard(self, encode_path: Path) -> None:
        writer = SparseShardWriter(
            output_dir=encode_path,
            vocab_size=8,
            rank=0,
            top_k=16,
            min_weight=0.05,
            exclude_output_ids=[1, 3],
            source_exclude_token_ids=[26, 27, 28],
            model_family="lens",
            compact_head_alignment="latent_cluster",
            output_token_aligned=False,
            shard_max_docs=32,
            value_dtype="float32",
        )
        writer.write_sparse_csr_batch(
            ["doc-1"],
            np.array([0, 2], dtype=np.int64),
            np.array([0, 5], dtype=np.int32),
            np.array([0.5, 1.5], dtype=np.float32),
            doc_group_ids=["parent-1"],
        )
        writer.finalize()

    def test_load_shard_manifest_preserves_output_space_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encode_path = Path(tmpdir) / "encode" / "lens_clustered"
            self._write_clustered_encode_shard(encode_path)

            shard_infos, metadata = load_shard_manifest(encode_path)

            self.assertEqual(len(shard_infos), 1)
            self.assertEqual(metadata["model_family"], "lens")
            self.assertEqual(metadata["compact_head_alignment"], "latent_cluster")
            self.assertFalse(metadata["output_token_aligned"])
            self.assertEqual(metadata["exclude_output_ids"], [1, 3])
            self.assertEqual(metadata["source_exclude_token_ids"], [26, 27, 28])
            self.assertTrue(metadata["has_group_ids"])
            self.assertEqual(shard_infos[0].group_ids_path.name, "shard_000000_group_ids.json")

    def test_index_script_metadata_records_cluster_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_name = "lens_clustered"
            encode_root = root / "encode"
            encode_path = encode_root / model_name
            self._write_clustered_encode_shard(encode_path)

            index_module = _load_index_script_module()
            cfg = OmegaConf.create(
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

            index_module.main.__wrapped__(cfg)

            metadata_path = root / "index" / model_name / "metadata.json"
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            self.assertEqual(metadata["model_family"], "lens")
            self.assertEqual(metadata["compact_head_alignment"], "latent_cluster")
            self.assertFalse(metadata["output_token_aligned"])
            self.assertEqual(metadata["exclude_output_ids"], [1, 3])
            self.assertEqual(metadata["source_exclude_token_ids"], [26, 27, 28])
            self.assertTrue(metadata["has_group_ids"])

            group_ids_path = root / "index" / model_name / "group_ids.json"
            with group_ids_path.open("r", encoding="utf-8") as handle:
                group_ids = json.load(handle)
            self.assertEqual(group_ids, ["parent-1"])

            loaded_index = load_inverted_index(root / "index" / model_name)
            self.assertEqual(loaded_index.group_ids, ["parent-1"])


if __name__ == "__main__":
    unittest.main()
