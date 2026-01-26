"""
Retriever models package.

To avoid circular imports, retriever classes are not imported at module level.
Import them explicitly when needed:
    from src.model.retriever.base import BaseRetriever
    from src.model.retriever.registry import RETRIEVER_REGISTRY
    from src.model.retriever.sparse.neural.splade import SPLADE
"""

from src.model.retriever.base import BaseRetriever
from src.model.retriever.registry import RETRIEVER_REGISTRY

__all__ = [
    "RETRIEVER_REGISTRY",
    "BaseRetriever",
]
