"""Backward-compatible shim for ANNA tokenizer implementation."""

from script.preprocess.anna.anna_tokenizer import AnnaTokenizer, AnnaTokenizerFast

__all__ = ["AnnaTokenizer", "AnnaTokenizerFast"]
