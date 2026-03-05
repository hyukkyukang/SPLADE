import unittest

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from config.path import ABS_CONFIG_DIR


class HydraConfigCompositionTest(unittest.TestCase):
    def _compose(self, *, config_name: str, overrides: list[str]) -> DictConfig:
        with initialize_config_dir(version_base=None, config_dir=ABS_CONFIG_DIR):
            return compose(config_name=config_name, overrides=overrides)

    def test_train_splade_v2_pp_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train",
            overrides=[
                "model=splade_v2_pp",
                "training=splade_v2_pp",
                "dataset@train_dataset=msmarco_spladev3_scores",
                "dataset@val_dataset=msmarco_spladev3_scores",
            ],
        )
        self.assertEqual(str(cfg.model.name), "splade_v2_pp")
        self.assertEqual(str(cfg.training.name), "splade_v2_pp")
        self.assertFalse(bool(cfg.training.disable_compile_for_validation))
        self.assertEqual(str(cfg.training.torch_compile_validation_mode), "default")
        self.assertIn("train_dataset", cfg)
        self.assertIn("val_dataset", cfg)

    def test_train_embeddinggemma_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train_embeddinggemma_splade_v2_pp",
            overrides=[
                "model=splade_v2_pp_embeddinggemma_300m_lsr",
                "training=splade_v2_pp_embeddinggemma_300m",
                "dataset@train_dataset=msmarco_spladev3_scores",
                "dataset@val_dataset=msmarco_spladev3_scores",
            ],
        )
        self.assertEqual(str(cfg.model.name), "splade_v2_pp_embeddinggemma_300m_lsr")
        self.assertEqual(str(cfg.training.name), "splade_v2_pp_embeddinggemma_300m")
        self.assertIn("train_dataset", cfg)
        self.assertIn("val_dataset", cfg)

    def test_validation_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="validation",
            overrides=[
                "model=splade_v2_pp",
                "training=splade_v2_pp",
                "dataset@train_dataset=msmarco_spladev3_scores",
                "dataset@val_dataset=msmarco_spladev3_scores",
            ],
        )
        self.assertEqual(str(cfg.model.name), "splade_v2_pp")
        self.assertEqual(str(cfg.training.name), "splade_v2_pp")
        self.assertIn("validation", cfg)

    def test_evaluation_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=splade_v2_pp",
                "dataset=msmarco_spladev3_scores",
            ],
        )
        self.assertEqual(str(cfg.model.name), "splade_v2_pp")
        self.assertEqual(str(cfg.evaluation.type), "retrieval")

    def test_nanobeir_evaluation_config_uses_nanobeir_experiment(self) -> None:
        cfg = self._compose(
            config_name="evaluate_nanobeir",
            overrides=["model=splade_v2_pp"],
        )
        self.assertEqual(str(cfg.mlflow.experiment_name), "NanoBEIR")
        self.assertTrue(bool(cfg.mlflow.enabled))

    def test_mteb_evaluation_config_uses_mteb_experiment(self) -> None:
        cfg = self._compose(
            config_name="evaluate_mteb",
            overrides=["model=splade_v2_pp"],
        )
        self.assertEqual(str(cfg.mlflow.experiment_name), "MTEB")
        self.assertTrue(bool(cfg.mlflow.enabled))


if __name__ == "__main__":
    unittest.main()
