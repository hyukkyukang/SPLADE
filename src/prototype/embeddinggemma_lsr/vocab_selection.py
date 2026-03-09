import math
from collections import Counter

SOURCE_TOKEN: str = "token"


def generic_unigram_df_penalty(
    *,
    term: str,
    term_df: int,
    doc_count: int,
    start_ratio: float,
    end_ratio: float,
    min_multiplier: float,
    penalty_power: float,
) -> float:
    parts: list[str] = [part for part in str(term).split() if part]
    if len(parts) != 1:
        return 1.0
    if int(doc_count) <= 0:
        return 1.0
    ratio: float = float(term_df) / float(doc_count)
    start: float = float(start_ratio)
    end: float = float(end_ratio)
    if ratio <= start:
        return 1.0
    if end <= start:
        return float(min_multiplier)
    clipped_ratio: float = min(max(ratio, start), end)
    progress: float = (clipped_ratio - start) / (end - start)
    shaped_progress: float = math.pow(progress, float(penalty_power))
    return 1.0 - shaped_progress * (1.0 - float(min_multiplier))


def phrase_cohesion_score(
    *,
    term: str,
    df_counter: Counter[str],
    doc_count: int,
    method: str,
) -> float | None:
    parts: list[str] = [part for part in str(term).split() if part]
    if len(parts) < 2 or int(doc_count) <= 0:
        return None
    term_df: int = int(df_counter.get(term, 0))
    if term_df <= 0:
        return None

    term_prob: float = float(term_df) / float(doc_count)
    if term_prob <= 0.0:
        return None

    part_probs: list[float] = []
    part: str
    for part in parts:
        part_df: int = int(df_counter.get(part, 0))
        if part_df <= 0:
            return None
        prob: float = float(part_df) / float(doc_count)
        if prob <= 0.0:
            return None
        part_probs.append(prob)

    denominator: float = 1.0
    prob: float
    for prob in part_probs:
        denominator *= prob
    if denominator <= 0.0:
        return None

    pmi: float = math.log(term_prob / denominator)
    if str(method).lower() == "pmi":
        return pmi
    normalizer: float = -math.log(term_prob)
    if normalizer <= 0.0:
        return None
    return pmi / normalizer


def resolve_term_source_boost(
    *,
    term: str,
    source_df_counter: dict[str, Counter[str]],
    source_boosts: dict[str, float],
) -> float:
    weighted_boost: float = 0.0
    contribution_total: int = 0
    source: str
    for source, boost in source_boosts.items():
        source_df: int = int(source_df_counter.get(source, Counter()).get(term, 0))
        if source_df <= 0:
            continue
        weighted_boost += float(source_df) * float(boost)
        contribution_total += int(source_df)
    if contribution_total <= 0:
        return float(source_boosts.get(SOURCE_TOKEN, 1.0))
    return weighted_boost / float(contribution_total)
