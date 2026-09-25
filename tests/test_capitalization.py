import statistics

import pytest

from conftest import reference_surprisal
from maze_distractors.capitalization import (
    CARRIERS,
    capital_preferences,
    capitalized_shares,
    is_often_capitalized,
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
def test_a_proper_noun_passes_both_signals(share, score, listed):
    assert is_often_capitalized(share, score) is listed
