import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.transformers import resolve_model_name_or_path


class TransformersSourceResolutionTest(unittest.TestCase):
    def test_resolve_model_name_or_path_uses_filesystem_fallback_without_checkpoint_load(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="source_resolution_") as tmp:
            model_root = Path(tmp) / "data" / "model"
            trained_dir = model_root / "trained_anna_base_hf"
            fallback_dir = model_root / "anna_base_hf"
            trained_dir.mkdir(parents=True, exist_ok=True)
            fallback_dir.mkdir(parents=True, exist_ok=True)
            (fallback_dir / "config.json").write_text(
                json.dumps({"model_type": "bert"}),
                encoding="utf-8",
            )

            with patch(
                "src.utils.transformers.torch.load",
                side_effect=AssertionError(
                    "resolve_model_name_or_path should not deserialize checkpoints"
                ),
            ):
                resolved = resolve_model_name_or_path(str(trained_dir))

            self.assertEqual(resolved, str(fallback_dir))


if __name__ == "__main__":
    unittest.main()
