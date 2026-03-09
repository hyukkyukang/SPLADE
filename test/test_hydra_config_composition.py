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
        self.assertEqual(str(cfg.training.mlflow.experiment_name), "Train-SPLADE")

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

    def test_train_splade_v2_pp_hard_config_composes(self) -> None:
        cfg = self._compose(
            config_name="train_splade_v2_pp_hard",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "splade_v2_pp")
        self.assertEqual(str(cfg.training.name), "splade_v2_pp_hard")
        self.assertEqual(str(cfg.train_dataset.name), "msmarco_hard_negatives")
        self.assertEqual(str(cfg.training.loss.type), "in_batch_plus_pairwise")
        self.assertAlmostEqual(float(cfg.training.loss.in_batch_weight), 1.0)
        self.assertAlmostEqual(float(cfg.training.loss.pairwise_weight), 1.0)

    def test_train_splade_v2_pp_sigmoid_hard_config_composes(self) -> None:
        cfg = self._compose(
            config_name="train_splade_v2_pp_sigmoid_hard",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "splade_v2_pp")
        self.assertEqual(str(cfg.training.name), "splade_v2_pp_sigmoid_hard")
        self.assertEqual(str(cfg.train_dataset.name), "msmarco_hard_negatives")
        self.assertEqual(str(cfg.training.loss.type), "sigmoid_pairwise_hard")
        self.assertAlmostEqual(
            float(cfg.training.loss.sigmoid.init_logit_scale), 2.302585093
        )
        self.assertAlmostEqual(float(cfg.training.loss.sigmoid.max_bias), -5.0)

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

    def test_default_msmarco_evaluation_uses_validation_qrels(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[],
        )
        self.assertEqual(str(cfg.dataset.name), "msmarco")
        self.assertEqual(str(cfg.dataset.type), "beir")
        self.assertEqual(str(cfg.dataset.qrels_hf_split), "validation")
        self.assertEqual(str(cfg.mlflow.experiment_name), "Eval-MSMARCO")
        self.assertTrue(bool(cfg.mlflow.enabled))

    def test_nanobeir_evaluation_config_uses_nanobeir_experiment(self) -> None:
        cfg = self._compose(
            config_name="evaluate_nanobeir",
            overrides=["model=splade_v2_pp"],
        )
        self.assertEqual(str(cfg.mlflow.experiment_name), "NanoBEIR")
        self.assertTrue(bool(cfg.mlflow.enabled))

    def test_mteb_evaluation_config_uses_eval_mteb_experiment(self) -> None:
        cfg = self._compose(
            config_name="evaluate_mteb",
            overrides=["model=splade_v2_pp"],
        )
        self.assertEqual(str(cfg.mlflow.experiment_name), "Eval-MTEB")
        self.assertTrue(bool(cfg.mlflow.enabled))


if __name__ == "__main__":
    unittest.main()
