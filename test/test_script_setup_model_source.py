import logging
import unittest

from omegaconf import OmegaConf

from src.utils.script_setup import resolve_model_source


class ResolveModelSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger('test_script_setup_model_source')

    def test_nanobeir_can_use_model_huggingface_name_without_testing_hf_model_path(self) -> None:
        cfg = OmegaConf.create(
            {
                'model': {'huggingface_name': 'yibinlei/LENS-d4000'},
                'testing': {'hf_model_path': None, 'checkpoint_path': None},
                'nanobeir': {'use_huggingface_model': True},
            }
        )

        resolved = resolve_model_source(
            cfg,
            logger=self.logger,
            set_nanobeir_flag=True,
        )

        self.assertEqual(resolved.model.huggingface_name, 'yibinlei/LENS-d4000')
        self.assertTrue(bool(resolved.nanobeir.use_huggingface_model))

    def test_checkpoint_or_explicit_hf_path_is_still_required_outside_nanobeir_mode(self) -> None:
        cfg = OmegaConf.create(
            {
                'model': {'huggingface_name': 'yibinlei/LENS-d4000'},
                'testing': {'hf_model_path': None, 'checkpoint_path': None},
                'nanobeir': {'use_huggingface_model': True},
            }
        )

        with self.assertRaisesRegex(ValueError, 'testing.checkpoint_path must be set'):
            resolve_model_source(cfg, logger=self.logger, set_nanobeir_flag=False)


if __name__ == '__main__':
    unittest.main()
