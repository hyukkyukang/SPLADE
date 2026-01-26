from __future__ import annotations

from src.model.retriever.base import BaseRetriever
from src.model.retriever.registry import RETRIEVER_REGISTRY


# Register SPLADE under the canonical config name.
@RETRIEVER_REGISTRY.register("splade")
class SPLADE(BaseRetriever):
    """SPLADE retriever registered for config usage."""
