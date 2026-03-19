"""Backward-compatible wrapper for the ANNA tokenizer implementation."""

from src.tokenization.anna_tokenizer import AnnaTokenizer, AnnaTokenizerFast

__all__ = ["AnnaTokenizer", "AnnaTokenizerFast"]
