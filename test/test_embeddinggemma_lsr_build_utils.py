import argparse
import sys
import types

from src.prototype.embeddinggemma_lsr.artifacts import (
    load_vocab_artifacts,
    resolve_term_stats_cache_path,
    write_json,
    write_text_lines,
)
from src.utils.cli_config import (
    apply_config_overrides,
    parser_default_values,
    resolve_torch_device,
    resolve_torch_dtype,
)
from src.prototype.embeddinggemma_lsr import vocab_linguistics
from src.prototype.embeddinggemma_lsr.vocab_audit import audit_vocab_stats
from src.prototype.embeddinggemma_lsr.vocab_filtering import (
    strict_post_selection_cleanup_reason,
)


def test_apply_config_overrides_preserves_non_default_cli_values(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="default-output")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("alpha: 0.2\noutput_dir: cfg-output\n", encoding="utf-8")

    args = parser.parse_args([])
    args.config = str(config_path)
    args.alpha = 0.3

    updated = apply_config_overrides(args, defaults=parser_default_values(parser))

    assert updated.alpha == 0.3
    assert updated.output_dir == "cfg-output"


def test_vocab_artifact_helpers_round_trip(tmp_path) -> None:
    artifact_dir = tmp_path / "vocab"
    write_text_lines(artifact_dir / "v_target.txt", ["alpha", "beta"])
    write_json(artifact_dir / "df_map.json", {"alpha": 10, "beta": 5}, sort_keys=True)

    vocab, df_map = load_vocab_artifacts(artifact_dir)

    assert vocab == ["alpha", "beta"]
    assert df_map == {"alpha": 10, "beta": 5}


def test_resolve_term_stats_cache_path_defaults_to_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    cache_path = resolve_term_stats_cache_path(
        output_dir=output_dir,
        configured_path=None,
    )
    assert cache_path == output_dir / "term_statistics.pkl"


def test_torch_helpers_accept_expected_aliases() -> None:
    assert str(resolve_torch_dtype("float16")) == "torch.float16"
    assert str(resolve_torch_dtype("bfloat16")) == "torch.bfloat16"
    assert str(resolve_torch_dtype("float32")) == "torch.float32"
    assert str(resolve_torch_device("cpu")) == "cpu"


def test_strict_post_selection_cleanup_reason_flags_short_unigram() -> None:
    reason = strict_post_selection_cleanup_reason(
        term="dc",
        drop_short_alpha_unigrams=True,
        short_alpha_unigram_max_len=2,
        short_alpha_unigram_whitelist={"kg", "km"},
        drop_about_numeric_phrases=True,
        drop_leading_numeric_function_phrases=True,
        drop_trailing_function_word_phrases=True,
        trailing_function_words={"of"},
        drop_abbreviation_heavy_phrases=True,
        abbreviation_phrase_whitelist=set(),
        drop_artifact_substrings=True,
        artifact_substrings={"uplog"},
    )
    assert reason == "short_alpha_unigram"


def test_vocab_audit_reports_expected_patterns() -> None:
    report = audit_vocab_stats(
        {
            "summary": {
                "doc_count": 100,
                "candidate_terms": 10,
                "strict_post_selection_cleanup": {"enabled": True},
            },
            "selected_terms": [
                {"term": "alpha", "pos_tag": "NOUN"},
                {"term": "kg", "pos_tag": "NOUN"},
                {"term": "about 10 minutes", "pos_tag": "NOUN"},
                {"term": "r and d", "pos_tag": "NOUN"},
                {"term": "city of", "pos_tag": "NOUN"},
                {"term": "2010 census", "pos_tag": "NOUN"},
            ],
        }
    )

    assert report["selected_terms"] == 6
    assert report["short_alpha_unigrams"]["count"] == 1
    assert report["about_numeric_phrases"]["count"] == 1
    assert report["abbreviation_heavy_phrases"]["count"] == 1
    assert report["trailing_function_word_phrases"]["count"] == 1
    assert report["year_terms"]["count"] == 1


def test_vocab_linguistics_numeric_and_phrase_head_helpers() -> None:
    assert (
        vocab_linguistics.numeric_term_quality_reason(
            term="10th century",
            max_tokens=3,
        )
        == "mixed_alphanumeric"
    )
    assert (
        vocab_linguistics.numeric_term_quality_reason(
            term="ten century",
            max_tokens=3,
        )
        == "noncanonical_numeric_phrase"
    )
    assert (
        vocab_linguistics.numeric_term_quality_reason(
            term="10 11 12",
            max_tokens=3,
        )
        is None
    )
    assert (
        vocab_linguistics.infer_phrase_head_universal_pos(
            [("very", "ADV"), ("old", "ADJ"), ("houses", "NOUN")]
        )
        == "NOUN"
    )


def test_vocab_linguistics_normalize_noun_forms_with_hybrid_agreement(
    monkeypatch,
) -> None:
    class FakeLemmatizer:
        def lemmatize(self, token: str, _pos: str) -> str:
            mapping = {
                "dogs": "dog",
                "children": "child",
            }
            return mapping.get(token, token)

    def fake_pos_tag_sents(sequences, tagset="universal"):
        assert tagset == "universal"
        tagged_sequences = []
        for sequence in sequences:
            if sequence == ["smart", "dogs"]:
                tagged_sequences.append([("smart", "ADJ"), ("dogs", "NOUN")])
            else:
                tagged_sequences.append([(token, "NOUN") for token in sequence])
        return tagged_sequences

    monkeypatch.setitem(
        sys.modules,
        "nltk",
        types.SimpleNamespace(pos_tag_sents=fake_pos_tag_sents),
    )
    monkeypatch.setattr(
        vocab_linguistics,
        "get_wordnet_lemmatizer",
        lambda: FakeLemmatizer(),
    )

    normalized_map, stats = (
        vocab_linguistics.normalize_noun_forms_with_hybrid_agreement(
            terms=["dogs", "children", "smart dogs", "river"],
            term_sources={},
            pos_batch_size=8,
            skip_entity_backed=False,
            include_phrases=True,
            exception_words=set(),
        )
    )

    assert normalized_map == {
        "dogs": "dog",
        "children": "child",
        "smart dogs": "smart dog",
    }
    assert stats["normalized_terms"] == 3
    assert stats["normalized_unigrams"] == 2
    assert stats["normalized_phrases"] == 1
