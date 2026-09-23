"""Whether a lowercase word is mainly a proper noun, from two signals that
fail in different places. Used by scripts/build_proper_noun_list.py to
build data/proper_nouns.txt.

The share of a word's occurrences that are capitalized in SUBTLEX-US
(Brysbaert & New, 2009) is the direct measure, but it also runs high for
words that open sentences ("additionally", "nope"). A language model's
preference for the capitalized form in the middle of a sentence does not,
but it cannot tell "jack" from "sponge": wherever a bare name fits, a bare
singular noun does not, so the lower-case form loses for grammar's sake.
A word has to pass both; one SUBTLEX-US lacks has to pass the model's test
alone, by twice the margin.
"""

import csv
import statistics
from collections.abc import Sequence
from pathlib import Path

from maze_distractors.surprisal import Scorer

# Capitalized more often than not, against 5-30% for an ordinary noun.
MIN_CAPITALIZED_SHARE = 0.5
# The model finds the capitalized form at least 8 times as likely.
MIN_MODEL_SCORE = 3.0
# For a word SUBTLEX-US lacks, the model has to be twice as sure.
MIN_MODEL_SCORE_ALONE = 6.0

# Mid-sentence, so a capital is not the sentence's; subject and object
# slots both, since names turn up in either.
CARRIERS = (
    "Did you know that",
    "Just yesterday,",
    "I'm going to talk to",
    "Have you seen",
    "Have you been to",
)


def capitalized_shares(subtlex_file: Path) -> dict[str, float]:
    """For each word of SUBTLEX-US's tab-separated text version, the share
    of its occurrences that do not start with a lowercase letter."""
    with subtlex_file.open(encoding="latin-1", newline="") as f:
        return {
            row["Word"].lower(): 1 - int(row["FREQlow"]) / int(row["FREQcount"])
            for row in csv.DictReader(f, delimiter="\t")
        }


def capital_preferences(
    scorer: Scorer, words: Sequence[str]
) -> dict[str, float]:
    """For each lowercase word, how many bits less surprising the model
    finds it capitalized than as written, averaged over CARRIERS."""
    contexts = [scorer.context_for(carrier) for carrier in CARRIERS]
    lower = scorer.score_exactly(contexts, words)
    capitalized = scorer.score_exactly(
        contexts, [word.capitalize() for word in words]
    )
    return {
        word: statistics.mean(a - b for a, b in zip(low, cap, strict=True))
        for word, low, cap in zip(words, lower, capitalized, strict=True)
    }


def is_likely_proper_noun(
    capitalized_share: float | None, model_score: float
) -> bool:
    """``capitalized_share`` is None for a word SUBTLEX-US does not have."""
    if capitalized_share is None:
        return model_score >= MIN_MODEL_SCORE_ALONE
    return (
        capitalized_share >= MIN_CAPITALIZED_SHARE
        and model_score >= MIN_MODEL_SCORE
    )
