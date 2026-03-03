import os
import unittest

from src.utils.script_setup import _resolve_tagged_log_dir


class TaggedLogDirResolverTest(unittest.TestCase):
    def test_resolve_tagged_log_dir_with_tag(self) -> None:
        resolved = _resolve_tagged_log_dir("log/train/splade_v2", "exp_a")
        self.assertEqual(
            resolved,
            os.path.join("log/train/splade_v2", "exp_a"),
        )

    def test_resolve_tagged_log_dir_without_tag_uses_no_tag(self) -> None:
        resolved = _resolve_tagged_log_dir("log/train/splade_v2", None)
        self.assertEqual(
            resolved,
            os.path.join("log/train/splade_v2", "no_tag"),
        )

    def test_resolve_tagged_log_dir_blank_tag_uses_no_tag(self) -> None:
        resolved = _resolve_tagged_log_dir("log/train/splade_v2", "   ")
        self.assertEqual(
            resolved,
            os.path.join("log/train/splade_v2", "no_tag"),
        )


if __name__ == "__main__":
    unittest.main()
