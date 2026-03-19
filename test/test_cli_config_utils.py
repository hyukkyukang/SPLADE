import argparse
import tempfile
import unittest
from pathlib import Path

from src.utils.cli_config import (
    apply_config_overrides,
    parser_default_values,
    resolve_torch_device,
    resolve_torch_dtype,
)


class CliConfigUtilsTest(unittest.TestCase):
    def test_apply_config_overrides_preserves_non_default_cli_values(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--output-dir", type=str, default="default-output")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "alpha: 0.2\noutput_dir: cfg-output\n", encoding="utf-8"
            )

            args = parser.parse_args([])
            args.config = str(config_path)
            args.alpha = 0.3

            updated = apply_config_overrides(
                args, defaults=parser_default_values(parser)
            )

        self.assertEqual(updated.alpha, 0.3)
        self.assertEqual(updated.output_dir, "cfg-output")

    def test_torch_helpers_accept_expected_aliases(self) -> None:
        self.assertEqual(str(resolve_torch_dtype("float16")), "torch.float16")
        self.assertEqual(str(resolve_torch_dtype("bfloat16")), "torch.bfloat16")
        self.assertEqual(str(resolve_torch_dtype("float32")), "torch.float32")
        self.assertEqual(str(resolve_torch_device("cpu")), "cpu")


if __name__ == "__main__":
    unittest.main()
