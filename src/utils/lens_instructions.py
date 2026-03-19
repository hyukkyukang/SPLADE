from omegaconf import DictConfig

from src.data.lens_formatting import resolve_instruction_text
from src.utils.normalize import normalize_optional_str

_GENERIC_RETRIEVAL_PROMPTS: dict[str, str] = {
    "question": "Represent this question for retrieving relevant documents.",
    "query": "Represent this query for retrieving relevant documents.",
    "retrieval": "Represent this query for retrieving relevant documents.",
    "retrieval_query": "Represent this query for retrieving relevant documents.",
    "search_query": "Represent this query for retrieving relevant documents.",
    "web_search_query": "Represent this query for retrieving relevant documents.",
}


def _normalize_prompt_name(prompt_name: str | None) -> str | None:
    normalized: str | None = normalize_optional_str(prompt_name)
    if normalized is None:
        return None
    return normalized.lower().replace("-", "_").replace(" ", "_")


def resolve_benchmark_instruction(
    model_cfg: DictConfig | None,
    *,
    prompt_name: str | None = None,
    prompt: str | None = None,
) -> str:
    explicit_prompt: str | None = normalize_optional_str(prompt)
    if explicit_prompt is not None:
        return explicit_prompt

    normalized_prompt_name: str | None = _normalize_prompt_name(prompt_name)
    if normalized_prompt_name is not None:
        mapped_prompt: str | None = _GENERIC_RETRIEVAL_PROMPTS.get(
            normalized_prompt_name
        )
        if mapped_prompt is not None:
            return mapped_prompt

    return resolve_instruction_text(model_cfg)
