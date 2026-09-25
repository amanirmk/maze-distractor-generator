import statistics

import pytest

from conftest import reference_surprisal
from maze_distractors.capitalization import (
    CARRIERS,
    capital_preferences,
    capitalized_shares,
    is_mainly_a_name,
    is_mainly_an_abbreviation,
    is_often_capitalized,
    lowercase_per_million,
)


def test_the_capitalized_share_comes_from_subtlex_counts(tmp_path):
    subtlex = tmp_path / "subtlex.txt"
    subtlex.write_text(
        "Word\tFREQcount\tCDcount\tFREQlow\n"
        "the\t1000\t10\t900\n"
        "Jack\t200\t10\t6\n"
    )  # fmt: skip
    assert capitalized_shares(subtlex) == pytest.approx(
        {"the": 0.1, "jack": 0.97}
    )
    # FREQlow over SUBTLEX-US's 51 million words.
    assert lowercase_per_million(subtlex) == pytest.approx(
        {"the": 900 / 51, "jack": 6 / 51}
    )


def test_the_model_score_is_bits_saved_by_the_capital_over_carriers(scorer):
    scores = capital_preferences(scorer, ["dog", "house"])
    assert list(scores) == ["dog", "house"]
    expected = statistics.mean(
        reference_surprisal(scorer, carrier, "house")
        - reference_surprisal(scorer, carrier, "House")
        for carrier in CARRIERS
    )
    assert scores["house"] == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    ("share", "score", "listed"),
    [
        (0.97, 4.9, True),  # jack
        (0.24, 5.6, False),  # sponge: the model alone would list it
        (0.98, 0.2, False),  # additionally: capitalized for opening sentences
        (None, 7.6, True),  # waldo: not in SUBTLEX-US, the model is sure
        (None, 4.0, False),
    ],
)
def test_an_often_capitalized_word_passes_both_signals(share, score, listed):
    assert is_often_capitalized(share, score) is listed


@pytest.mark.parametrize(
    ("share", "lowercase", "left_out"),
    [
        (0.99, 0.22, True),  # josh
        (0.97, 7.14, False),  # jack: a word too
        (0.50, 14.6, False),  # sue
        (0.92, 2.33, False),  # eve
        (0.92, 0.90, False),  # heather: a plant as well, capitalized less
        (None, None, False),  # not in SUBTLEX-US: not measured, kept
    ],
)
def test_a_name_goes_when_readers_know_it_only_as_a_name(
    share, lowercase, left_out
):
    assert is_mainly_a_name(share, lowercase) is left_out


@pytest.mark.parametrize(
    ("share", "lowercase", "left_out"),
    [
        (0.80, 5.0, True),  # pa
        (0.91, 0.0, True),  # rev
        (0.46, 101.0, False),  # ma: mother
        (0.97, 88.0, False),  # oh: capitalized to open sentences
        (0.09, 5.0, False),  # rep
    ],
)
def test_an_abbreviation_goes_when_readers_know_it_only_as_one(
    share, lowercase, left_out
):
    assert is_mainly_an_abbreviation(share, lowercase) is left_out
