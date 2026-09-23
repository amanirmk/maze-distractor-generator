"""The scoring path on the default model. Slow, and skipped unless the
model is already in the local HuggingFace cache:

uv run pytest -m slow
"""

import math

import pytest

from conftest import reference_surprisal
from maze_distractors.cli import DEFAULT_MODEL
from maze_distractors.surprisal import Scorer

pytestmark = pytest.mark.slow
WORDS = [
    "The",
    "office",
    "worker",
    "eventually",
    "remembered",
    "his",
    "password",
    "expired.",
]


@pytest.fixture(scope="module")
def real_scorer(monkeypatch_module) -> Scorer:
    monkeypatch_module.setenv("HF_HUB_OFFLINE", "1")
    try:
        return Scorer.from_pretrained(DEFAULT_MODEL, device="cpu")
    except OSError:
        pytest.skip(f"{DEFAULT_MODEL} is not in the local cache")


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def test_gpt2_starts_sequences_with_endoftext(real_scorer):
    assert real_scorer.start_token == "<|endoftext|>"


def test_sentence_and_candidate_surprisals_match_the_reference(real_scorer):
    scored = real_scorer.score_words(WORDS)
    for i, word in enumerate(scored, start=1):
        assert word.surprisal == pytest.approx(
            reference_surprisal(real_scorer, " ".join(WORDS[:i]), WORDS[i]),
            abs=1e-3,
        )
    # One token and several, after a short context and a long one.
    candidates = ["summers", "anyway", "unhesitatingly", "xylophonist"]
    contexts = [scored[0].context, scored[-1].context]
    scores = real_scorer.score_candidates(contexts, candidates, [math.inf] * 2)
    for candidate, row in zip(candidates, scores, strict=True):
        for context, score in zip(contexts, row, strict=True):
            assert score == pytest.approx(
                reference_surprisal(real_scorer, context.text, candidate),
                abs=1e-3,
            )
