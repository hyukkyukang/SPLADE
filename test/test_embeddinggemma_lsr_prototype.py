import math
from types import SimpleNamespace

import torch

from src.prototype.embeddinggemma_lsr.losses import (
    compute_ranking_metrics,
    flops_regularization,
    info_nce_in_batch,
)
from src.prototype.embeddinggemma_lsr.model import (
    EmbeddingGemmaLSRModel,
    apply_projection_initialization,
)


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        _ = attention_mask
        values: torch.Tensor = input_ids.to(dtype=torch.float32)
        hidden: torch.Tensor = torch.stack([values, values + 1.0], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, *_args, **_kwargs) -> None:
        raise NotImplementedError


def test_embeddinggemma_lsr_forward_masked_max_pooling() -> None:
    model = EmbeddingGemmaLSRModel(
        backbone=DummyBackbone(),
        target_vocab=["t0", "t1"],
    )
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.projection.bias.zero_()

    input_ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    reps: torch.Tensor = model(input_ids=input_ids, attention_mask=attention_mask)

    expected = torch.tensor([[math.log1p(2.0), math.log1p(3.0)]], dtype=torch.float32)
    assert torch.allclose(reps, expected, atol=1e-6)


def test_apply_projection_initialization_sets_parameters() -> None:
    model = EmbeddingGemmaLSRModel(
        backbone=DummyBackbone(),
        target_vocab=["a", "b", "c"],
    )
    weights = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    biases = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    apply_projection_initialization(model, weights=weights, biases=biases)
    assert torch.allclose(model.projection.weight, weights)
    assert torch.allclose(model.projection.bias, biases)


def test_losses_and_metrics_smoke() -> None:
    query_reps = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    doc_reps = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    loss, logits = info_nce_in_batch(query_reps, doc_reps, temperature=1.0)
    assert float(loss.item()) > 0.0
    assert tuple(logits.shape) == (2, 2)

    flops = flops_regularization(query_reps)
    assert float(flops.item()) > 0.0

    scores = torch.tensor([[2.0, 1.0], [0.5, 3.0]], dtype=torch.float32)
    positives = torch.tensor([0, 1], dtype=torch.long)
    metrics = compute_ranking_metrics(scores, positive_indices=positives, k_values=(10,))

    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0
