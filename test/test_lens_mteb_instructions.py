"""Spot-check per-task instruction prompts against the official LENS repo."""

from __future__ import annotations

from src.utils.lens_mteb_instructions import task_to_instruction


_EXPECTED = {
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query.",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question.",
    "ArguAna": "Given a claim, find documents that refute the claim.",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question.",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question.",
    "TRECCOVID": "Given a query, retrieve documents that answer the query.",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question.",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim.",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question.",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia.",
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim.",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim.",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question.",
    "Banking77Classification": "Given a online banking query, find the corresponding intents.",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset.",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment.",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles.",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles.",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum.",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers.",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum.",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet.",
}


def test_known_task_prompts_have_period_terminator() -> None:
    for task, expected in _EXPECTED.items():
        actual = task_to_instruction(task, is_query=True)
        assert actual == expected, (
            f"Task {task!r}: got {actual!r}, expected {expected!r}"
        )


def test_retrieval_corpus_side_empty() -> None:
    # For retrieval tasks the corpus instruction is the empty string (no period).
    assert task_to_instruction("MSMARCO", is_query=False) == ""
    assert task_to_instruction("NFCorpus", is_query=False) == ""
    assert task_to_instruction("HotpotQA", is_query=False) == ""


def test_unknown_task_returns_safe_default() -> None:
    # An obviously bogus task name should not raise, just return "" or a default.
    result = task_to_instruction("__SomeUnknownTaskZZZ__", is_query=True)
    assert isinstance(result, str)
