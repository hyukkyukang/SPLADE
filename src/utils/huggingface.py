import os


def resolve_hf_token() -> str | None:
    token: str | None = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if token is None:
        return None
    normalized: str = str(token).strip()
    return normalized or None


__all__ = ["resolve_hf_token"]
