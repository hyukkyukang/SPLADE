import importlib
import sys
import unittest
from contextlib import contextmanager
from typing import Iterator

from import_stubs import (
    install_fake_numba,
    install_fake_pytorch_lightning_utilities,
)

install_fake_numba()
install_fake_pytorch_lightning_utilities()


@contextmanager
def _isolated_module_prefix(prefix: str) -> Iterator[None]:
    preserved = {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(f"{prefix}.")
    }
    for name in list(preserved):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(preserved)


class LazyImportBoundariesTest(unittest.TestCase):
    def test_importing_search_sparsify_does_not_import_retrieval_side_modules(self) -> None:
        with _isolated_module_prefix("src.search"), _isolated_module_prefix("src.index"):
            importlib.import_module("src.search.sparsify")

            self.assertIn("src.search", sys.modules)
            self.assertIn("src.search.sparsify", sys.modules)
            self.assertNotIn("src.search.retrieval", sys.modules)
            self.assertNotIn("src.search.scoring", sys.modules)
            self.assertNotIn("src.index.async_writer", sys.modules)

    def test_importing_index_sparse_does_not_import_async_writer(self) -> None:
        with _isolated_module_prefix("src.index"):
            importlib.import_module("src.index.sparse")

            self.assertIn("src.index", sys.modules)
            self.assertIn("src.index.sparse", sys.modules)
            self.assertNotIn("src.index.async_writer", sys.modules)

    def test_search_package_attribute_loads_target_module_lazily(self) -> None:
        with _isolated_module_prefix("src.search"), _isolated_module_prefix("src.index"):
            search_module = importlib.import_module("src.search")

            self.assertNotIn("src.search.buffers", sys.modules)
            self.assertNotIn("src.search.retrieval", sys.modules)

            _ = search_module.prepare_score_buffers

            self.assertIn("src.search.buffers", sys.modules)
            self.assertNotIn("src.search.retrieval", sys.modules)

    def test_index_package_attribute_loads_sparse_without_async_writer(self) -> None:
        with _isolated_module_prefix("src.index"):
            index_module = importlib.import_module("src.index")

            self.assertNotIn("src.index.sparse", sys.modules)
            self.assertNotIn("src.index.async_writer", sys.modules)

            _ = index_module.resolve_numpy_dtype

            self.assertIn("src.index.sparse", sys.modules)
            self.assertNotIn("src.index.async_writer", sys.modules)

    def test_importing_dataset_package_does_not_eagerly_load_dataset_modules(self) -> None:
        with _isolated_module_prefix("src.data.dataset"):
            importlib.import_module("src.data.dataset")

            self.assertIn("src.data.dataset", sys.modules)
            self.assertNotIn("src.data.dataset.base", sys.modules)
            self.assertNotIn("src.data.dataset.msmarco", sys.modules)

    def test_importing_retriever_package_does_not_eagerly_load_registry_modules(self) -> None:
        with _isolated_module_prefix("src.model.retriever"):
            importlib.import_module("src.model.retriever")

            self.assertIn("src.model.retriever", sys.modules)
            self.assertNotIn("src.model.retriever.base", sys.modules)
            self.assertNotIn("src.model.retriever.registry", sys.modules)

    def test_importing_tokenization_package_does_not_eagerly_import_script_module(self) -> None:
        with _isolated_module_prefix("src.tokenization"), _isolated_module_prefix(
            "script.preprocess.anna"
        ):
            importlib.import_module("src.tokenization")

            self.assertIn("src.tokenization", sys.modules)
            self.assertNotIn("src.tokenization.anna_tokenizer", sys.modules)
            self.assertNotIn("script.preprocess.anna.anna_tokenizer", sys.modules)


if __name__ == "__main__":
    unittest.main()
