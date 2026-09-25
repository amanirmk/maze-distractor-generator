"""Whether a lowercase word is often capitalized -- a name, a title, a
place, a brand, a word fiction capitalizes ("the Academy") -- from two
signals that fail in different places. Used by
scripts/build_often_capitalized_list.py to build
data/often_capitalized.txt, whose words are held to their surprisal
threshold capitalized as well as shown (generation.py).

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
import hashlib
import io
import statistics
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

from maze_distractors.surprisal import Scorer

SUBTLEX_URL = (
    "https://www.ugent.be/pp/experimentele-psychologie/en/research/"
    "documents/subtlexus/subtlexus2.zip"
)
SUBTLEX_SHA256 = (
    "67e595da1b399d2a21e25a1466a8d9f242f21a219c6bbbd3e089c4779ad83856"
)
SUBTLEX_MEMBER = "SUBTLEXus74286wordstextversion.txt"
# Words in SUBTLEX-US, for counts per million.
SUBTLEX_MILLIONS = 51.0

# Capitalized in at least 40% of occurrences, clearly above the 5-30% of
# an ordinary noun. The list only adds a check, so a word that is also
# ordinary costs little; one read as a name and missed costs more ("ma",
# 46%, was chosen where "Ma" was no surprise).
MIN_CAPITALIZED_SHARE = 0.4
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


def download_subtlex(directory: Path) -> Path:
    """SUBTLEX-US's text version, downloaded into ``directory`` and checked
    against the copy the shipped lists were built from."""
    with urllib.request.urlopen(SUBTLEX_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SUBTLEX_SHA256:
        raise SystemExit(
            f"{SUBTLEX_URL} is not the file the lists were built from."
        )
    archive = zipfile.ZipFile(io.BytesIO(data))
    return Path(archive.extract(SUBTLEX_MEMBER, directory))


def lowercase_per_million(subtlex_file: Path) -> dict[str, float]:
    """For each word of SUBTLEX-US's text version, how often it occurs in
    lower case, per million words: how often it is an ordinary word."""
    with subtlex_file.open(encoding="latin-1", newline="") as f:
        return {
            row["Word"].lower(): int(row["FREQlow"]) / SUBTLEX_MILLIONS
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


def is_often_capitalized(
    capitalized_share: float | None, model_score: float
) -> bool:
    """``capitalized_share`` is None for a word SUBTLEX-US does not have."""
    if capitalized_share is None:
        return model_score >= MIN_MODEL_SCORE_ALONE
    return (
        capitalized_share >= MIN_CAPITALIZED_SHARE
        and model_score >= MIN_MODEL_SCORE
    )


# A first name is left out when readers meet it almost only as a name:
# capitalized nearly always, and in lower case less than once per million
# words. "josh", "tony" and "terry" go; "sue", "drew", "eve", "jack" and
# "frank", ordinary words too, stay.
NAME_MIN_CAPITALIZED_SHARE = 0.95
NAME_MAX_LOWERCASE_PER_MILLION = 1.0
# An abbreviation is left out when it is capitalized more often than not
# and rare in lower case: "pa" and "rev" go; "ma" (mother), "rep", "oh",
# "in" and "or" stay.
ABBREVIATION_MIN_CAPITALIZED_SHARE = 0.5
ABBREVIATION_MAX_LOWERCASE_PER_MILLION = 10.0


def is_mainly_a_name(
    capitalized_share: float | None, lowercase: float | None
) -> bool:
    """For a word that is a common first name: whether readers know it
    only as the name. A word SUBTLEX-US lacks is not left out."""
    if capitalized_share is None or lowercase is None:
        return False
    return (
        capitalized_share >= NAME_MIN_CAPITALIZED_SHARE
        and lowercase < NAME_MAX_LOWERCASE_PER_MILLION
    )


def is_mainly_an_abbreviation(
    capitalized_share: float | None, lowercase: float | None
) -> bool:
    """For a word that is a state code or an abbreviated title: whether
    readers know it only as the abbreviation."""
    if capitalized_share is None or lowercase is None:
        return False
    return (
        capitalized_share >= ABBREVIATION_MIN_CAPITALIZED_SHARE
        and lowercase < ABBREVIATION_MAX_LOWERCASE_PER_MILLION
    )
