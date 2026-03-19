import sys
import types
from functools import wraps
from importlib.machinery import ModuleSpec
from typing import Any


def install_fake_hydra() -> None:
    try:
        import hydra  # noqa: F401
    except ImportError:
        hydra_module = types.ModuleType("hydra")

        def _main(*args: Any, **kwargs: Any):
            def decorator(func):
                @wraps(func)
                def wrapped(*func_args: Any, **func_kwargs: Any):
                    return func(*func_args, **func_kwargs)

                return wrapped

            return decorator

        hydra_module.main = _main
        sys.modules["hydra"] = hydra_module


def install_fake_mlflow() -> None:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        mlflow_module = types.ModuleType("mlflow")
        mlflow_module.__spec__ = ModuleSpec("mlflow", loader=None)

        class _Dataset:
            def __init__(
                self,
                *,
                name: str,
                digest: str,
                source_type: str,
                source: str | None,
                schema: str | None = None,
                profile: str | None = None,
            ) -> None:
                self.name = name
                self.digest = digest
                self.source_type = source_type
                self.source = source
                self.schema = schema
                self.profile = profile

        class _InputTag:
            def __init__(self, key: str, value: str) -> None:
                self.key = key
                self.value = value

        class _DatasetInput:
            def __init__(self, dataset: _Dataset, tags: list[_InputTag]) -> None:
                self.dataset = dataset
                self.tags = tags

        class _LoggedModelOutput:
            def __init__(self, model_id: str, step: int) -> None:
                self.model_id = model_id
                self.step = step

        class _MlflowClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = args, kwargs

        def _from_pandas(df: Any, name: str, source: str | None = None) -> Any:
            _ = df

            class _DatasetWrapper:
                def to_dict(self) -> dict[str, Any]:
                    return {
                        "name": name,
                        "digest": f"digest-{name}",
                        "source_type": "pandas",
                        "source": source,
                        "schema": None,
                        "profile": None,
                    }

            return _DatasetWrapper()

        mlflow_module.data = types.SimpleNamespace(from_pandas=_from_pandas)
        mlflow_module.get_tracking_uri = lambda: None
        mlflow_module.set_tracking_uri = lambda uri: None
        mlflow_module.create_external_model = (
            lambda **kwargs: types.SimpleNamespace(model_id="external-model-id")
        )
        mlflow_module.active_run = lambda: None
        mlflow_module.start_run = lambda **kwargs: None
        mlflow_module.end_run = lambda **kwargs: None

        entities_module = types.ModuleType("mlflow.entities")
        entities_module.__spec__ = ModuleSpec("mlflow.entities", loader=None)
        entities_module.Dataset = _Dataset
        entities_module.DatasetInput = _DatasetInput
        entities_module.InputTag = _InputTag

        logged_model_output_module = types.ModuleType(
            "mlflow.entities.logged_model_output"
        )
        logged_model_output_module.__spec__ = ModuleSpec(
            "mlflow.entities.logged_model_output", loader=None
        )
        logged_model_output_module.LoggedModelOutput = _LoggedModelOutput

        tracking_module = types.ModuleType("mlflow.tracking")
        tracking_module.__spec__ = ModuleSpec("mlflow.tracking", loader=None)
        tracking_module.MlflowClient = _MlflowClient

        mlflow_module.entities = entities_module
        mlflow_module.tracking = tracking_module

        sys.modules["mlflow"] = mlflow_module
        sys.modules["mlflow.entities"] = entities_module
        sys.modules[
            "mlflow.entities.logged_model_output"
        ] = logged_model_output_module
        sys.modules["mlflow.tracking"] = tracking_module


def install_fake_pytorch_lightning_utilities() -> None:
    try:
        from pytorch_lightning.utilities import rank_zero_only  # noqa: F401
    except Exception:
        pytorch_lightning_module = sys.modules.get("pytorch_lightning")
        if pytorch_lightning_module is None:
            pytorch_lightning_module = types.ModuleType("pytorch_lightning")
            pytorch_lightning_module.__spec__ = ModuleSpec(
                "pytorch_lightning", loader=None
            )
            sys.modules["pytorch_lightning"] = pytorch_lightning_module

        utilities_module = types.ModuleType("pytorch_lightning.utilities")
        utilities_module.__spec__ = ModuleSpec(
            "pytorch_lightning.utilities", loader=None
        )

        def rank_zero_only(fn):
            @wraps(fn)
            def wrapped(*args: Any, **kwargs: Any):
                return fn(*args, **kwargs)

            return wrapped

        utilities_module.rank_zero_only = rank_zero_only
        pytorch_lightning_module.utilities = utilities_module
        sys.modules["pytorch_lightning.utilities"] = utilities_module


def install_fake_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError:
        pandas_module = types.ModuleType("pandas")
        pandas_module.__spec__ = ModuleSpec("pandas", loader=None)

        class DataFrame:
            def __init__(self, rows: Any) -> None:
                self.rows = rows

        pandas_module.DataFrame = DataFrame
        sys.modules["pandas"] = pandas_module


def install_fake_numba() -> None:
    try:
        import numba  # noqa: F401
    except ImportError:
        numba_module = types.ModuleType("numba")
        numba_module.__spec__ = ModuleSpec("numba", loader=None)

        def _njit(func=None, *args: Any, **kwargs: Any):
            if func is not None and callable(func):
                return func

            def decorator(inner):
                return inner

            return decorator

        numba_module.njit = _njit
        sys.modules["numba"] = numba_module


def install_fake_sentence_transformers() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        sentence_transformers_module = types.ModuleType("sentence_transformers")
        sentence_transformers_module.__spec__ = ModuleSpec(
            "sentence_transformers", loader=None
        )

        class SparseEncoder:
            def __init__(
                self,
                modules: Any | None = None,
                similarity_fn_name: str = "dot",
            ) -> None:
                self.modules = modules or []
                self.similarity_fn_name = similarity_fn_name

            def to(self, device: Any) -> "SparseEncoder":
                self.device = device
                return self

            def eval(self) -> "SparseEncoder":
                return self

            @staticmethod
            def sparsity(embeddings: Any) -> dict[str, float]:
                if hasattr(embeddings, "is_sparse") and embeddings.is_sparse:
                    embeddings = embeddings.to_dense()
                if not hasattr(embeddings, "numel") or embeddings.numel() == 0:
                    return {
                        "active_dims": 0.0,
                        "sparsity_ratio": 1.0,
                    }
                active_dims = float((embeddings != 0).sum(dim=1).float().mean().item())
                width = float(embeddings.shape[1]) if int(embeddings.shape[1]) > 0 else 1.0
                return {
                    "active_dims": active_dims,
                    "sparsity_ratio": max(0.0, 1.0 - (active_dims / width)),
                }

        models_module = types.ModuleType("sentence_transformers.models")
        models_module.__spec__ = ModuleSpec(
            "sentence_transformers.models", loader=None
        )

        class Normalize:
            def __call__(self, tensor: Any) -> Any:
                return tensor

        models_module.Normalize = Normalize

        sparse_encoder_module = types.ModuleType("sentence_transformers.sparse_encoder")
        sparse_encoder_module.__spec__ = ModuleSpec(
            "sentence_transformers.sparse_encoder", loader=None
        )

        sparse_encoder_models_module = types.ModuleType(
            "sentence_transformers.sparse_encoder.models"
        )
        sparse_encoder_models_module.__spec__ = ModuleSpec(
            "sentence_transformers.sparse_encoder.models", loader=None
        )

        class MLMTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = args, kwargs
                self.auto_model = types.SimpleNamespace(
                    load_state_dict=lambda state_dict, strict=False: types.SimpleNamespace(
                        missing_keys=[],
                        unexpected_keys=[],
                    )
                )

            def get_sentence_embedding_dimension(self) -> int:
                return 0

        class SpladePooling:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = args, kwargs

        sparse_encoder_models_module.MLMTransformer = MLMTransformer
        sparse_encoder_models_module.SpladePooling = SpladePooling

        evaluation_module = types.ModuleType(
            "sentence_transformers.sparse_encoder.evaluation"
        )
        evaluation_module.__spec__ = ModuleSpec(
            "sentence_transformers.sparse_encoder.evaluation", loader=None
        )

        class SparseNanoBEIREvaluator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = args, kwargs

        evaluation_module.SparseNanoBEIREvaluator = SparseNanoBEIREvaluator

        sentence_transformers_module.SparseEncoder = SparseEncoder
        sentence_transformers_module.models = models_module
        sentence_transformers_module.sparse_encoder = sparse_encoder_module
        sparse_encoder_module.models = sparse_encoder_models_module
        sparse_encoder_module.evaluation = evaluation_module

        sys.modules["sentence_transformers"] = sentence_transformers_module
        sys.modules["sentence_transformers.models"] = models_module
        sys.modules["sentence_transformers.sparse_encoder"] = sparse_encoder_module
        sys.modules[
            "sentence_transformers.sparse_encoder.models"
        ] = sparse_encoder_models_module
        sys.modules[
            "sentence_transformers.sparse_encoder.evaluation"
        ] = evaluation_module
