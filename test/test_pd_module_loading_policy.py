import unittest
from unittest.mock import Mock, patch

from omegaconf import DictConfig, OmegaConf

from src.data.pd_module.base import PDModule
from src.data.pd_module.encode import EncodePDModule
from src.data.pd_module.reranking import RerankingPDModule
from src.data.pd_module.train import TrainingPDModule
from src.data.pl_module.train import TrainDataModule


def build_dataset_cfg(*, hf_name: str | None = "dummy/hf") -> DictConfig:
    return OmegaConf.create(
        {
            "name": "dummy",
            "max_query_length": 8,
            "max_doc_length": 16,
            "max_padding": False,
            "num_positives": 1,
            "num_negatives": 1,
            "hf_name": hf_name,
            "query_corpus_hf_name": None,
        }
    )


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id: int = 0
        self.pad_token: str = "[PAD]"
        self.eos_token: str = "[EOS]"
        self.cls_token: str = "[CLS]"
        self.is_fast: bool = True


class DummyDataset:
    def __init__(self) -> None:
        self.events: list[str] = []
        self._meta_dataset: list[dict[str, str]] = [{"query_id": "q0"}]
        self._query_dataset: list[dict[str, str]] = [{"query_id": "q0", "query": "q"}]
        self._corpus_dataset: list[dict[str, str]] = [{"doc_id": "d0", "text": "d"}]

    def prepare_meta_dataset(self) -> None:
        self.events.append("prepare_meta_dataset")

    @property
    def meta_dataset(self) -> list[dict[str, str]]:
        self.events.append("meta_dataset")
        return self._meta_dataset

    @property
    def query_dataset(self) -> list[dict[str, str]]:
        self.events.append("query_dataset")
        return self._query_dataset

    @property
    def corpus_dataset(self) -> list[dict[str, str]]:
        self.events.append("corpus_dataset")
        return self._corpus_dataset

    @property
    def query_dataset_id_to_idx(self) -> dict[str, int]:
        self.events.append("query_dataset_id_to_idx")
        return {"q0": 0}

    @property
    def corpus_dataset_id_to_idx(self) -> dict[str, int]:
        self.events.append("corpus_dataset_id_to_idx")
        return {"d0": 0}


class PDModuleLoadingPolicyTest(unittest.TestCase):
    def test_base_prepare_data_only_prepares_meta_dataset(self) -> None:
        module = PDModule(
            cfg=build_dataset_cfg(),
            tokenizer=DummyTokenizer(),
            seed=123,
        )
        dataset = DummyDataset()
        module._dataset = dataset

        module.prepare_data()

        self.assertEqual(dataset.events, ["prepare_meta_dataset"])

    def test_training_setup_warms_query_corpus_and_id_maps(self) -> None:
        module = TrainingPDModule(
            cfg=build_dataset_cfg(),
            tokenizer=DummyTokenizer(),
            seed=123,
        )
        dataset = DummyDataset()
        module._dataset = dataset

        module.setup()

        self.assertIn("meta_dataset", dataset.events)
        self.assertIn("query_dataset", dataset.events)
        self.assertIn("corpus_dataset", dataset.events)
        self.assertIn("query_dataset_id_to_idx", dataset.events)
        self.assertIn("corpus_dataset_id_to_idx", dataset.events)

    def test_reranking_setup_warms_query_corpus_and_id_maps(self) -> None:
        module = RerankingPDModule(
            cfg=build_dataset_cfg(),
            tokenizer=DummyTokenizer(),
            seed=123,
        )
        dataset = DummyDataset()
        module._dataset = dataset

        module.setup()

        self.assertIn("meta_dataset", dataset.events)
        self.assertIn("query_dataset", dataset.events)
        self.assertIn("corpus_dataset", dataset.events)
        self.assertIn("query_dataset_id_to_idx", dataset.events)
        self.assertIn("corpus_dataset_id_to_idx", dataset.events)

    def test_encode_prepare_data_is_noop_and_setup_warms_only_corpus(self) -> None:
        module = EncodePDModule(
            cfg=build_dataset_cfg(),
            tokenizer=DummyTokenizer(),
            seed=123,
        )
        dataset = DummyDataset()
        module._dataset = dataset

        module.prepare_data()
        self.assertEqual(dataset.events, [])

        module.setup()

        self.assertIn("corpus_dataset", dataset.events)
        self.assertNotIn("meta_dataset", dataset.events)
        self.assertNotIn("query_dataset", dataset.events)
        self.assertNotIn("query_dataset_id_to_idx", dataset.events)
        self.assertNotIn("corpus_dataset_id_to_idx", dataset.events)

    @patch("src.data.pl_module.train.build_tokenizer")
    def test_train_data_module_calls_prepare_and_setup_for_both_datasets(
        self, tokenizer_loader: Mock
    ) -> None:
        tokenizer_loader.return_value = DummyTokenizer()
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model": {
                    "huggingface_name": "dummy/model",
                    "use_fast_tokenizer": True,
                    "trust_remote_code": False,
                    "require_fast_tokenizer": False,
                },
                "train_dataset": build_dataset_cfg(),
                "val_dataset": build_dataset_cfg(),
                "training": {
                    "distill": {"enabled": False, "fail_on_missing": True},
                    "num_workers": 0,
                    "prefetch_factor": 1,
                    "use_cpu": True,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                },
            }
        )
        data_module = TrainDataModule(cfg=cfg)
        self.assertFalse(data_module.prepare_data_per_node)
        train_dataset_mock: Mock = Mock()
        val_dataset_mock: Mock = Mock()
        data_module.__dict__["train_dataset"] = train_dataset_mock
        data_module.__dict__["val_dataset"] = val_dataset_mock

        data_module.prepare_data()
        data_module.setup()

        train_dataset_mock.prepare_data.assert_called_once()
        val_dataset_mock.prepare_data.assert_called_once()
        train_dataset_mock.setup.assert_called_once()
        val_dataset_mock.setup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
