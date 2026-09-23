import pytest

from maze_distractors.punctuation import copy_punctuation, strip_punctuation


@pytest.mark.parametrize(
    ("word", "core"),
    [("cat", "cat"), ('"Hello,', "Hello"), ("dogs'", "dogs"), ("--", "")],
)
def test_strip_punctuation(word, core):
    assert strip_punctuation(word) == core


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("expired.", "summers."),
        ('"Hello,', '"Summers,'),
        ("NASA", "SUMMERS"),
        ("don't", "summers"),
        ("I", "summers"),  # one capital letter is not a capitalized word
    ],
)
def test_copy_punctuation(word, expected):
    assert copy_punctuation(word, "summers") == expected
