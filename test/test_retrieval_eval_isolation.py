import unittest

from omegaconf import OmegaConf

from src.utils.evaluation_mode import enforce_retrieval_evaluation_isolation
from src.utils.logging import get_logger


class RetrievalEvalIsolationTest(unittest.TestCase):
    def test_forces_nanobeir_disabled_in_retrieval_eval(self) -> None:
        cfg = OmegaConf.create(
            {
                "evaluation": {"type": "retrieval"},
                "nanobeir": {"enabled": True},
            }
        )
        logger = get_logger("test.retrieval_eval_isolation")
        resolved_cfg = enforce_retrieval_evaluation_isolation(cfg, logger=logger)
        self.assertFalse(bool(resolved_cfg.nanobeir.enabled))

    def test_rejects_non_retrieval_mode(self) -> None:
        cfg = OmegaConf.create(
            {
                "evaluation": {"type": "validation"},
                "nanobeir": {"enabled": False},
            }
        )
        logger = get_logger("test.retrieval_eval_isolation.reject")
        with self.assertRaisesRegex(ValueError, "evaluation.type=retrieval"):
            enforce_retrieval_evaluation_isolation(cfg, logger=logger)


if __name__ == "__main__":
    unittest.main()
