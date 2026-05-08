import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.model.retriever.sparse.neural.splade import SpladeEncoder, SpladeModel
from src.utils.compact_head import OFFICIAL_LENS_HEAD_FILENAME


class _DummyHiddenOutput:
    def __init__(self, hidden_states: torch.Tensor) -> None:
        self.last_hidden_state: torch.Tensor = hidden_states


class _DummyHiddenModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size: int = int(hidden_size)
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> _DummyHiddenOutput:
        _ = attention_mask, return_dict, use_cache
        batch_size: int = int(input_ids.shape[0])
        seq_len: int = int(input_ids.shape[1])
        hidden = torch.ones((batch_size, seq_len, self.hidden_size), dtype=torch.float32)
        return _DummyHiddenOutput(hidden)


class _DummyModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits: torch.Tensor = logits


class _DummyCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            vocab_size=int(vocab_size),
            use_cache=False,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            bos_token_id=3,
            eos_token_id=4,
        )
        self.model = _DummyHiddenModel(hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, int(vocab_size), bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> _DummyModelOutput:
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=use_cache,
        ).last_hidden_state
        return _DummyModelOutput(self.lm_head(hidden))

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, module: nn.Module | None) -> None:
        self.lm_head = nn.Identity() if module is None else module


def _build_compact_payload(
    *,
    out_features: int,
    hidden_size: int,
    alignment: str | None,
    include_token_ids: bool,
    token_ids: list[int] | None = None,
) -> dict[str, torch.Tensor | list[int] | str]:
    payload: dict[str, torch.Tensor | list[int] | str] = {
        "weight": torch.ones((out_features, hidden_size), dtype=torch.float32),
        "bias": torch.zeros((out_features,), dtype=torch.float32),
    }
    if alignment is not None:
        payload["alignment"] = alignment
    if include_token_ids:
        payload["token_ids"] = (
            list(range(out_features)) if token_ids is None else list(token_ids)
        )
    return payload


class LensCompactHeadAlignmentTest(unittest.TestCase):
    def test_legacy_token_aligned_compact_head_keeps_token_mapping(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=32, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="compact_head_legacy_") as tmp:
            model_dir = Path(tmp)
            torch.save(
                _build_compact_payload(
                    out_features=4,
                    hidden_size=8,
                    alignment=None,
                    include_token_ids=True,
                ),
                model_dir / "splade_compact_head.pt",
            )

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                encoder = SpladeEncoder(
                    model_name=str(model_dir),
                    sparse_activation="log1p_relu",
                    huggingface_model_class="AutoModelForCausalLM",
                )

        self.assertTrue(encoder.output_token_aligned)
        self.assertEqual(encoder.compact_head_alignment, "token_ids")
        self.assertEqual(tuple(encoder.token_id_to_output_index.tolist()), (0, 1, 2, 3))

    def test_cluster_aligned_compact_head_skips_token_mapping(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=32, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="compact_head_cluster_") as tmp:
            model_dir = Path(tmp)
            torch.save(
                _build_compact_payload(
                    out_features=4,
                    hidden_size=8,
                    alignment="latent_cluster",
                    include_token_ids=True,
                ),
                model_dir / "splade_compact_head.pt",
            )

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                encoder = SpladeEncoder(
                    model_name=str(model_dir),
                    sparse_activation="log1p_relu",
                    huggingface_model_class="AutoModelForCausalLM",
                )

        self.assertFalse(encoder.output_token_aligned)
        self.assertEqual(encoder.compact_head_alignment, "latent_cluster")
        self.assertEqual(int(encoder.token_id_to_output_index.numel()), 0)
        mask = encoder.build_exclude_mask(torch.tensor([0, 1], dtype=torch.long))
        self.assertEqual(int(mask.numel()), 0)

    def test_token_aligned_compact_head_maps_token_ids_to_output_ids(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=64, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="compact_head_token_map_") as tmp:
            model_dir = Path(tmp)
            torch.save(
                _build_compact_payload(
                    out_features=3,
                    hidden_size=8,
                    alignment="token_ids",
                    include_token_ids=True,
                    token_ids=[10, 20, 30],
                ),
                model_dir / "splade_compact_head.pt",
            )

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                encoder = SpladeEncoder(
                    model_name=str(model_dir),
                    sparse_activation="log1p_relu",
                    huggingface_model_class="AutoModelForCausalLM",
                )

        resolved_output_ids = encoder.resolve_output_exclude_ids(
            torch.tensor([20, 999, 10], dtype=torch.long)
        )
        self.assertEqual(tuple(resolved_output_ids.tolist()), (0, 1))
        mask = encoder.build_exclude_mask(torch.tensor([20], dtype=torch.long))
        self.assertTrue(torch.equal(mask, torch.tensor([False, True, False])))

    def test_doc_only_rejects_cluster_aligned_compact_head(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=32, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="compact_head_doc_only_") as tmp:
            model_dir = Path(tmp)
            torch.save(
                _build_compact_payload(
                    out_features=4,
                    hidden_size=8,
                    alignment="latent_cluster",
                    include_token_ids=False,
                ),
                model_dir / "splade_compact_head.pt",
            )

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                with self.assertRaisesRegex(ValueError, "doc_only requires token-aligned"):
                    _ = SpladeModel(
                        family="lens",
                        model_name=str(model_dir),
                        huggingface_model_class="AutoModelForCausalLM",
                        query_pooling="max",
                        doc_pooling="max",
                        sparse_activation="log1p_relu",
                        doc_only=True,
                    )


    def test_official_lens_lm_head_file_loads_as_clustered_head(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=32, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="official_lens_local_") as tmp:
            model_dir = Path(tmp)
            torch.save(nn.Linear(8, 4, bias=True), model_dir / OFFICIAL_LENS_HEAD_FILENAME)

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value=str(model_dir),
            ):
                encoder = SpladeEncoder(
                    model_name=str(model_dir),
                    sparse_activation="log1p_relu",
                    huggingface_model_class="MistralBiForCausalLM",
                )

        self.assertFalse(encoder.output_token_aligned)
        self.assertEqual(encoder.compact_head_alignment, "latent_cluster")
        self.assertEqual(tuple(encoder.compact_head.weight.shape), (4, 8))

    def test_official_lens_hub_head_downloads_when_model_source_is_repo_id(self) -> None:
        dummy_model = _DummyCausalLM(vocab_size=32, hidden_size=8)
        with tempfile.TemporaryDirectory(prefix="official_lens_hub_") as tmp:
            artifact_path = Path(tmp) / OFFICIAL_LENS_HEAD_FILENAME
            torch.save(nn.Linear(8, 5, bias=False), artifact_path)

            with patch(
                "src.model.retriever.sparse.neural.splade.build_masked_lm_model",
                return_value=dummy_model,
            ), patch(
                "src.model.retriever.sparse.neural.splade.resolve_model_name_or_path",
                return_value="yibinlei/LENS-d4000",
            ), patch(
                "src.model.retriever.sparse.neural.splade._maybe_download_hf_artifact",
                return_value=artifact_path,
            ) as mocked_download:
                encoder = SpladeEncoder(
                    model_name="yibinlei/LENS-d4000",
                    sparse_activation="log1p_relu",
                    huggingface_model_class="MistralBiForCausalLM",
                )

        self.assertEqual(mocked_download.call_count, 1)
        self.assertFalse(encoder.output_token_aligned)
        self.assertEqual(tuple(encoder.compact_head.weight.shape), (5, 8))


if __name__ == "__main__":
    unittest.main()
