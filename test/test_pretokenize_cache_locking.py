import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.pd_module.train import TrainingPDModule
from test_train_pretokenize_cache import (
    CountingTokenizer,
    DummyTrainDataset,
    _build_cfg,
)


class PretokenizeCacheLockingTest(unittest.TestCase):
    def test_manifest_mismatch_raises_when_overwrite_disabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pretoken_manifest_mismatch_") as tmp_dir:
            cfg = _build_cfg(tmp_dir)
            tokenizer = CountingTokenizer()
            module = TrainingPDModule(
                cfg=cfg,
                tokenizer=tokenizer,
                seed=13,
                cache_namespace="train",
            )
            module._dataset = DummyTrainDataset()
            module.prepare_data()

            mismatch_cfg = _build_cfg(tmp_dir)
            mismatch_cfg.max_query_length = 9
            mismatch_module = TrainingPDModule(
                cfg=mismatch_cfg,
                tokenizer=CountingTokenizer(),
                seed=17,
                cache_namespace="train",
            )
            mismatch_module._dataset = DummyTrainDataset()
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                mismatch_module.prepare_data()

    def test_stale_lock_is_recovered_during_prepare_data(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pretoken_stale_lock_") as tmp_dir:
            cfg = _build_cfg(tmp_dir)
            tokenizer = CountingTokenizer()
            module = TrainingPDModule(
                cfg=cfg,
                tokenizer=tokenizer,
                seed=19,
                cache_namespace="train",
            )
            module._dataset = DummyTrainDataset()

            cache_dir: Path = Path(tmp_dir) / "train"
            cache_dir.mkdir(parents=True, exist_ok=True)
            lock_path: Path = cache_dir / "build.lock"
            lock_path.write_text("99999999", encoding="utf-8")

            module.prepare_data()
            self.assertTrue((cache_dir / "build.done").is_file())
            self.assertTrue((cache_dir / "manifest.json").is_file())

    def test_active_lock_wait_path_uses_wait_for_done(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pretoken_active_lock_") as tmp_dir:
            cfg = _build_cfg(tmp_dir)
            tokenizer = CountingTokenizer()
            module = TrainingPDModule(
                cfg=cfg,
                tokenizer=tokenizer,
                seed=23,
                cache_namespace="train",
            )
            module._dataset = DummyTrainDataset()

            cache_dir: Path = Path(tmp_dir) / "train"
            cache_dir.mkdir(parents=True, exist_ok=True)
            lock_path: Path = cache_dir / "build.lock"
            lock_path.write_text(str(os.getpid()), encoding="utf-8")

            with patch(
                "src.data.pd_module.pretokenize_lifecycle.wait_for_done"
            ) as wait_for_done_mock:
                module._build_or_validate_cache()
                wait_for_done_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
