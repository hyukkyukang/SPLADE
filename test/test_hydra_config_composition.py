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
        self.assertEqual(int(cfg.training.torch_compile_large_vocab_threshold), 30000)
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

    def test_train_mdlm_splade_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train_mdlm_splade",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "mdlm_splade_distilbert")
        self.assertEqual(str(cfg.model.family), "mdlm_splade")
        self.assertEqual(str(cfg.training.name), "mdlm_splade")
        self.assertEqual(str(cfg.training.torch_compile_mode), "default")
        self.assertEqual(int(cfg.training.torch_compile_large_vocab_threshold), 30000)
        self.assertTrue(bool(cfg.training.mdlm.enabled))
        self.assertAlmostEqual(float(cfg.training.mdlm.weight), 0.01)
        self.assertTrue(bool(cfg.training.torch_compile_train_core_when_possible))
        self.assertFalse(bool(cfg.training.find_unused_parameters))
        self.assertEqual(str(cfg.training.mdlm.doc_selection), "positives")
        self.assertEqual(int(cfg.training.mdlm.doc_chunk_size), 0)
        self.assertIn("train_dataset", cfg)
        self.assertIn("val_dataset", cfg)

    def test_pretrained_diffusion_splade_template_composes(self) -> None:
        cfg = self._compose(
            config_name="train",
            overrides=[
                "model=pretrained_diffusion_splade_template",
                "training=splade_v2",
                "dataset@train_dataset=msmarco",
                "dataset@val_dataset=msmarco_dev_small_negatives",
            ],
        )
        self.assertEqual(str(cfg.model.name), "pretrained_diffusion_splade_template")
        self.assertEqual(str(cfg.model.family), "pretrained_diffusion_splade")
        self.assertEqual(str(cfg.model.backbone_pretraining_type), "diffusion")
        self.assertEqual(str(cfg.model.tokenizer_name), "distilbert-base-uncased")
        self.assertTrue(bool(cfg.model.enforce_same_tokenizer_as_baseline))

    def test_train_pretrained_diffusion_splade_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train_pretrained_diffusion_splade",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "pretrained_diffusion_splade_udlm_lm1b")
        self.assertEqual(str(cfg.model.family), "pretrained_diffusion_splade")
        self.assertEqual(str(cfg.training.name), "pretrained_diffusion_splade")
        self.assertFalse(bool(cfg.training.mdlm.enabled))
        self.assertEqual(str(cfg.model.backbone_pretraining_type), "diffusion")
        self.assertEqual(str(cfg.model.huggingface_name), "kuleshov-group/udlm-lm1b")
        self.assertEqual(str(cfg.model.tokenizer_name), "bert-base-uncased")
        self.assertEqual(str(cfg.model.huggingface_model_class), "UDLMForMaskedLMCompat")
        self.assertFalse(bool(cfg.model.trust_remote_code))
        self.assertEqual(
            str(cfg.model.model_revision),
            "00dfee2a0578719ea93739884173d4393906a8fd",
        )
        self.assertEqual(int(cfg.train_dataset.max_doc_length), 128)
        self.assertEqual(int(cfg.val_dataset.max_doc_length), 128)
        self.assertIn("train_dataset", cfg)
        self.assertIn("val_dataset", cfg)

    def test_train_pretrained_diffusion_mdlm_splade_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train_pretrained_diffusion_mdlm_splade",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "pretrained_diffusion_splade_udlm_lm1b")
        self.assertEqual(str(cfg.model.family), "pretrained_diffusion_splade")
        self.assertEqual(str(cfg.training.name), "pretrained_diffusion_mdlm_splade")
        self.assertTrue(bool(cfg.training.mdlm.enabled))
        self.assertAlmostEqual(float(cfg.training.mdlm.weight), 0.01)
        self.assertTrue(bool(cfg.training.torch_compile_train_core_when_possible))
        self.assertEqual(str(cfg.training.mdlm.doc_selection), "positives")
        self.assertEqual(str(cfg.model.backbone_pretraining_type), "diffusion")
        self.assertEqual(str(cfg.model.huggingface_name), "kuleshov-group/udlm-lm1b")
        self.assertEqual(str(cfg.model.tokenizer_name), "bert-base-uncased")
        self.assertEqual(str(cfg.model.huggingface_model_class), "UDLMForMaskedLMCompat")
        self.assertFalse(bool(cfg.model.trust_remote_code))
        self.assertEqual(
            str(cfg.model.model_revision),
            "00dfee2a0578719ea93739884173d4393906a8fd",
        )
        self.assertEqual(int(cfg.train_dataset.max_doc_length), 128)
        self.assertEqual(int(cfg.val_dataset.max_doc_length), 128)
        self.assertIn("train_dataset", cfg)
        self.assertIn("val_dataset", cfg)

    def test_train_ordered_mask_slot_splade_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train_ordered_mask_slot_splade",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "ordered_mask_slot_splade_distilbert")
        self.assertEqual(str(cfg.model.family), "ordered_mask_slot_splade")
        self.assertEqual(str(cfg.training.name), "ordered_mask_slot_splade")
        self.assertEqual(int(cfg.model.num_mask_slots), 8)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_batch_size), 2048)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_log_interval), 100000)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_num_workers), 0)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_shards_per_worker), 4)
        self.assertTrue(bool(cfg.training.ordered_mask_slots.enabled))
        self.assertAlmostEqual(float(cfg.training.ordered_mask_slots.query_term_weight), 0.1)
        self.assertAlmostEqual(float(cfg.training.ordered_mask_slots.doc_term_weight), 0.1)
        self.assertFalse(bool(cfg.training.find_unused_parameters))
        self.assertTrue(bool(cfg.training.torch_compile_train_core_when_possible))
        self.assertEqual(int(cfg.train_dataset.max_query_length), 128)
        self.assertEqual(int(cfg.train_dataset.max_doc_length), 128)
        self.assertEqual(int(cfg.val_dataset.max_query_length), 128)
        self.assertEqual(int(cfg.val_dataset.max_doc_length), 128)
        self.assertEqual(int(cfg.training.val_check_interval_optimizer_steps), 5000)
        self.assertTrue(bool(cfg.training.validation_sparse_probe.enabled))
        self.assertEqual(int(cfg.training.validation_sparse_probe.num_pairs), 10)
        self.assertEqual(int(cfg.training.validation_sparse_probe.top_k_sparse), 20)
        self.assertEqual(int(cfg.training.validation_sparse_probe.top_k_slot), 10)
        self.assertEqual(
            list(cfg.training.validation_sparse_probe.probe_indices),
            [3817, 1431, 865, 970, 2550, 2800, 2209, 3278, 1128, 1795],
        )
        self.assertIn("train_dataset", cfg)
        self.assertIn("val_dataset", cfg)

    def test_train_pretrained_diffusion_ordered_mask_slot_splade_matrix_composes(self) -> None:
        cfg = self._compose(
            config_name="train_pretrained_diffusion_ordered_mask_slot_splade",
            overrides=[],
        )
        self.assertEqual(
            str(cfg.model.name),
            "pretrained_diffusion_ordered_mask_slot_splade_udlm_lm1b",
        )
        self.assertEqual(
            str(cfg.model.family),
            "pretrained_diffusion_ordered_mask_slot_splade",
        )
        self.assertEqual(
            str(cfg.training.name),
            "pretrained_diffusion_ordered_mask_slot_splade",
        )
        self.assertEqual(int(cfg.model.num_mask_slots), 8)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_batch_size), 2048)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_log_interval), 100000)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_num_workers), 0)
        self.assertEqual(int(cfg.model.ordered_mask_slots.idf_shards_per_worker), 4)
        self.assertTrue(bool(cfg.training.ordered_mask_slots.enabled))
        self.assertEqual(str(cfg.model.huggingface_name), "kuleshov-group/udlm-lm1b")
        self.assertEqual(str(cfg.model.tokenizer_name), "bert-base-uncased")
        self.assertEqual(str(cfg.model.huggingface_model_class), "UDLMForMaskedLMCompat")
        self.assertFalse(bool(cfg.model.trust_remote_code))
        self.assertTrue(bool(cfg.training.torch_compile_train_core_when_possible))
        self.assertFalse(bool(cfg.training.find_unused_parameters))
        self.assertEqual(int(cfg.training.val_check_interval_optimizer_steps), 5000)
        self.assertTrue(bool(cfg.training.validation_sparse_probe.enabled))
        self.assertEqual(int(cfg.training.validation_sparse_probe.num_pairs), 10)
        self.assertEqual(int(cfg.training.validation_sparse_probe.top_k_sparse), 20)
        self.assertEqual(int(cfg.training.validation_sparse_probe.top_k_slot), 10)
        self.assertEqual(
            list(cfg.training.validation_sparse_probe.probe_indices),
            [3817, 1431, 865, 970, 2550, 2800, 2209, 3278, 1128, 1795],
        )
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
            float(cfg.training.loss.sigmoid.init_logit_scale), -6.907755279
        )
        self.assertAlmostEqual(float(cfg.training.loss.sigmoid.init_bias), -8.0)
        self.assertAlmostEqual(float(cfg.training.loss.sigmoid.max_bias), -5.0)
        self.assertEqual(float(cfg.training.val_check_interval), 1.0)
        self.assertIsNone(cfg.training.val_check_interval_optimizer_steps)
        self.assertEqual(int(cfg.training.torch_compile_large_vocab_threshold), 30000)

    def test_train_splade_v3_patent_hard_config_composes(self) -> None:
        cfg = self._compose(
            config_name="train_splade_v3_patent_hard",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "splade_v3_naver")
        self.assertEqual(str(cfg.model.huggingface_name), "naver/splade-v3")
        self.assertEqual(str(cfg.training.name), "splade_v3_patent_hard")
        self.assertEqual(str(cfg.train_dataset.name), "patent_10k_hard_negatives")
        self.assertEqual(str(cfg.val_dataset.name), "patent_10k_hard_negatives")
        self.assertEqual(str(cfg.training.loss.type), "in_batch_plus_pairwise")
        self.assertFalse(bool(cfg.training.distill.enabled))
        self.assertEqual(str(cfg.training.regularization.query_type), "l1")
        self.assertEqual(str(cfg.training.regularization.doc_type), "flops")
        self.assertAlmostEqual(float(cfg.training.regularization.query_weight), 0.01)
        self.assertAlmostEqual(float(cfg.training.regularization.doc_weight), 0.02)
        self.assertEqual(int(cfg.training.regularization.schedule_steps), 30000)
        self.assertEqual(int(cfg.train_dataset.max_query_length), 64)
        self.assertEqual(int(cfg.train_dataset.max_doc_length), 256)
        self.assertEqual(int(cfg.train_dataset.num_negatives), 8)
        self.assertEqual(
            str(cfg.train_dataset.negative_sampling.strategy), "topk_plus_random"
        )
        self.assertEqual(str(cfg.val_dataset.negative_sampling.strategy), "topk")

    def test_dense_encode_and_evaluate_configs_compose(self) -> None:
        encode_cfg = self._compose(
            config_name="encode",
            overrides=[
                "model=all_minilm_l6_v2",
                "dataset=patent_us_corpus_small",
            ],
        )
        self.assertEqual(str(encode_cfg.model.family), "dense")
        self.assertEqual(str(encode_cfg.model.huggingface_model_class), "AutoModel")
        self.assertEqual(str(encode_cfg.model.similarity), "cosine")

        evaluate_cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=all_minilm_l6_v2",
                "dataset=patent_us_small_eval",
                "testing=patent_us_small_eval",
            ],
        )
        self.assertTrue(bool(evaluate_cfg.testing.faiss_use_gpu))
        self.assertFalse(bool(evaluate_cfg.testing.faiss_gpu_required))

    def test_dpr_bilingual_dense_config_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=dpr_bilingual_negative1_ko_en",
                "dataset=patent_us_small_eval",
                "testing=patent_us_small_eval",
            ],
        )
        self.assertEqual(str(cfg.model.family), "dense")
        self.assertEqual(str(cfg.model.dense_architecture), "dpr_biencoder")
        self.assertEqual(str(cfg.model.query_pooling), "cls")
        self.assertEqual(str(cfg.model.doc_pooling), "cls")
        self.assertEqual(int(cfg.model.bert_config.hidden_size), 768)
        self.assertEqual(int(cfg.model.bert_config.num_hidden_layers), 12)

    def test_dpr_bilingual_passage_eval_config_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=dpr_bilingual_negative1_ko_en",
                "dataset=patent_us_small_eval_dpr",
                "testing=patent_us_small_eval_dpr",
            ],
        )
        self.assertEqual(str(cfg.dataset.name), "patent_us_small_eval_dpr")
        self.assertEqual(str(cfg.dataset.corpus_id_column), "passage_id")
        self.assertEqual(str(cfg.dataset.corpus_group_id_column), "parent_doc_id")
        self.assertEqual(str(cfg.testing.result_group_key), "group_id")
        self.assertEqual(int(cfg.testing.group_candidate_pool), 4096)

    def test_train_splade_v3_patent_20k_hard_config_composes(self) -> None:
        cfg = self._compose(
            config_name="train_splade_v3_patent_20k_hard",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "splade_v3_naver")
        self.assertEqual(str(cfg.model.huggingface_name), "naver/splade-v3")
        self.assertEqual(str(cfg.training.name), "splade_v3_patent_hard")
        self.assertEqual(str(cfg.train_dataset.name), "patent_20k_hard_negatives")
        self.assertEqual(str(cfg.val_dataset.name), "patent_20k_hard_negatives")
        self.assertEqual(str(cfg.required_train_dataset_name), "patent_20k_hard_negatives")
        self.assertEqual(str(cfg.training.regularization.query_type), "l1")
        self.assertEqual(str(cfg.training.regularization.doc_type), "flops")
        self.assertAlmostEqual(float(cfg.training.regularization.query_weight), 0.01)
        self.assertAlmostEqual(float(cfg.training.regularization.doc_weight), 0.02)
        self.assertEqual(int(cfg.training.regularization.schedule_steps), 30000)
        self.assertEqual(str(cfg.train_dataset.hf_name), "Hyukkyu/patent-20k")
        self.assertEqual(str(cfg.train_dataset.negative_sampling.strategy), "topk_plus_random")
        self.assertEqual(str(cfg.val_dataset.negative_sampling.strategy), "topk")
        self.assertFalse(bool(cfg.nanobeir.enabled))

    def test_train_splade_v3_patent_us_in_batch_config_composes(self) -> None:
        cfg = self._compose(
            config_name="train_splade_v3_patent_us_in_batch",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "splade_v3_naver")
        self.assertEqual(str(cfg.model.huggingface_name), "naver/splade-v3")
        self.assertEqual(str(cfg.training.name), "splade_v3_patent_in_batch")
        self.assertEqual(str(cfg.train_dataset.name), "patent_us_in_batch")
        self.assertEqual(str(cfg.val_dataset.name), "patent_us_in_batch")
        self.assertEqual(str(cfg.required_train_dataset_name), "patent_us_in_batch")
        self.assertEqual(str(cfg.training.loss.type), "in_batch")
        self.assertEqual(int(cfg.train_dataset.max_query_length), 64)
        self.assertEqual(int(cfg.train_dataset.max_doc_length), 512)
        self.assertEqual(int(cfg.train_dataset.num_negatives), 0)
        self.assertEqual(str(cfg.train_dataset.hf_name), "parquet")
        self.assertEqual(
            str(cfg.train_dataset.hf_data_files.train),
            "data/patent/train/usc102103_in_batch_metadata.parquet",
        )
        self.assertEqual(
            str(cfg.train_dataset.corpus_hf_data_files.train),
            ".cache/hf/patent-us-corpus/patent_us_docs_slice*.parquet",
        )
        self.assertFalse(bool(cfg.nanobeir.enabled))

    def test_patent_20k_dataset_override_composes(self) -> None:
        cfg = self._compose(
            config_name="train_splade_v3_patent_hard",
            overrides=[
                "dataset@train_dataset=patent_20k_hard_negatives",
                "dataset@val_dataset=patent_20k_hard_negatives",
                "required_train_dataset_name=patent_20k_hard_negatives",
            ],
        )
        self.assertEqual(str(cfg.train_dataset.name), "patent_20k_hard_negatives")
        self.assertEqual(str(cfg.val_dataset.name), "patent_20k_hard_negatives")
        self.assertEqual(str(cfg.train_dataset.type), "patent_10k_hard_negatives")
        self.assertEqual(str(cfg.train_dataset.hf_name), "Hyukkyu/patent-20k")

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

    def test_encode_patent_us_corpus_small_config_composes(self) -> None:
        cfg = self._compose(
            config_name="encode",
            overrides=["dataset=patent_us_corpus_small"],
        )
        self.assertEqual(str(cfg.dataset.name), "patent_us_corpus_small")
        self.assertEqual(str(cfg.dataset.type), "corpus_only")
        self.assertEqual(str(cfg.dataset.query_corpus_hf_name), "parquet")
        self.assertIsNone(cfg.dataset.query_subset_name)
        self.assertIsNone(cfg.dataset.corpus_subset_name)
        self.assertEqual(
            str(cfg.dataset.query_corpus_hf_data_files.train),
            ".cache/hf/patent-us-corpus-small/data/*.parquet",
        )
        self.assertEqual(str(cfg.dataset.corpus_text_template), "patent_document_v1")
        self.assertEqual(
            list(cfg.dataset.corpus_additional_text_columns),
            ["claims", "description"],
        )

    def test_encode_patent_us_claim_passages_small_config_composes(self) -> None:
        cfg = self._compose(
            config_name="encode",
            overrides=[
                "model=dpr_bilingual_negative1_ko_en",
                "dataset=patent_us_claim_passages_small",
            ],
        )
        self.assertEqual(str(cfg.dataset.name), "patent_us_claim_passages_small")
        self.assertEqual(str(cfg.dataset.corpus_id_column), "passage_id")
        self.assertEqual(str(cfg.dataset.corpus_group_id_column), "parent_doc_id")
        self.assertIsNone(cfg.dataset.corpus_text_template)

    def test_evaluate_patent_us_small_config_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "dataset=patent_us_small_eval",
                "testing=patent_us_small_eval",
            ],
        )
        self.assertEqual(str(cfg.dataset.name), "patent_us_small_eval")
        self.assertEqual(str(cfg.testing.name), "patent_us_small_eval")
        self.assertEqual(str(cfg.dataset.query_corpus_hf_name), "parquet")
        self.assertIsNone(cfg.dataset.query_subset_name)
        self.assertEqual(
            str(cfg.dataset.query_hf_data_files.test),
            "data/eval/patent_us_small/queries.parquet",
        )
        self.assertEqual(
            str(cfg.dataset.corpus_hf_data_files.train),
            ".cache/hf/patent-us-corpus-small/data/*.parquet",
        )
        self.assertEqual(
            str(cfg.dataset.qrels_hf_data_files.test),
            "data/eval/patent_us_small/qrels.parquet",
        )
        self.assertEqual(list(cfg.testing.k_list), [1, 5, 10, 16, 32, 63, 150, 300])
        self.assertEqual(list(cfg.testing.metric_families), ["MRR", "Recall"])

    def test_evaluate_patent_us_small_dpr_config_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=dpr_bilingual_negative1_ko_en",
                "dataset=patent_us_small_eval_dpr",
                "testing=patent_us_small_eval_dpr",
            ],
        )
        self.assertEqual(str(cfg.dataset.query_hf_data_files.test), "data/eval/patent_us_small_dpr_plain_claims/queries.parquet")
        self.assertEqual(str(cfg.dataset.corpus_hf_data_files.train), "data/corpus/patent_us_claim_passages_small/passages.parquet")
        self.assertEqual(str(cfg.dataset.qrels_hf_data_files.test), "data/eval/patent_us_small_dpr_plain_claims/qrels.parquet")
        self.assertEqual(str(cfg.testing.result_group_key), "group_id")

    def test_evaluate_patent_us_small_passage_grouped_sparse_config_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=splade_v3_naver",
                "dataset=patent_us_small_eval_passage_grouped",
                "testing=patent_us_small_eval_passage_grouped",
            ],
        )
        self.assertEqual(
            str(cfg.dataset.query_hf_data_files.test),
            "data/eval/patent_us_small_dpr_plain_claims/queries.parquet",
        )
        self.assertEqual(
            str(cfg.dataset.corpus_hf_data_files.train),
            "data/corpus/patent_us_claim_passages_small/passages.parquet",
        )
        self.assertEqual(
            str(cfg.dataset.qrels_hf_data_files.test),
            "data/eval/patent_us_small_dpr_plain_claims/qrels.parquet",
        )
        self.assertEqual(str(cfg.testing.result_group_key), "group_id")
        self.assertEqual(int(cfg.testing.group_candidate_pool), 4096)
        self.assertTrue(bool(cfg.testing.exclude_self_match))

    def test_evaluate_patent_us_small_dpr_gcdpr_config_composes(self) -> None:
        cfg = self._compose(
            config_name="evaluate",
            overrides=[
                "model=dpr_bilingual_negative1_ko_en",
                "dataset=patent_us_small_eval_dpr_gcdpr",
                "testing=patent_us_small_eval_dpr_gcdpr",
            ],
        )
        self.assertEqual(str(cfg.dataset.query_hf_data_files.test), "data/eval/patent_us_small_dpr_gcdpr/queries.parquet")
        self.assertEqual(str(cfg.dataset.qrels_hf_data_files.test), "data/eval/patent_us_small_dpr_gcdpr/qrels.parquet")
        self.assertEqual(str(cfg.testing.result_group_key), "group_id")
        self.assertEqual(int(cfg.testing.group_candidate_pool), 200)
        self.assertEqual(int(cfg.testing.search_top_k), 200)
        self.assertFalse(bool(cfg.testing.exclude_self_match))
        self.assertTrue(bool(cfg.testing.faiss_gpu_shard))
        self.assertEqual(list(cfg.testing.k_list), [1, 5, 10, 16, 32, 64, 150, 1000, 3000, 10000])
        self.assertEqual(list(cfg.testing.metric_families), ["Success"])

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

    def test_patent_dpr_gcdpr_evaluation_config_uses_patent_experiment(self) -> None:
        cfg = self._compose(
            config_name="evaluate_patent_dpr_gcdpr",
            overrides=[],
        )
        self.assertEqual(str(cfg.model.name), "dpr_bilingual_negative1_ko_en")
        self.assertEqual(str(cfg.dataset.name), "patent_us_small_eval_dpr_gcdpr")
        self.assertEqual(str(cfg.testing.name), "patent_us_small_eval_dpr_gcdpr")
        self.assertEqual(str(cfg.mlflow.experiment_name), "Eval-Patent-DPR")
        self.assertTrue(bool(cfg.mlflow.enabled))
        self.assertTrue(bool(cfg.mlflow.log_artifacts))
        self.assertEqual(str(cfg.mlflow.tags.task), "patent_document_retrieval")
        self.assertEqual(str(cfg.mlflow.tags.domain), "patent")
        self.assertEqual(str(cfg.mlflow.tags.protocol), "gcdpr_proxy")


if __name__ == "__main__":
    unittest.main()
