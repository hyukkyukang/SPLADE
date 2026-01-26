from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Build a Hugging Face tokenizer with a safe pad token."""
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name, use_fast=True
    )
    # Ensure padding works for models without an explicit pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    return tokenizer
