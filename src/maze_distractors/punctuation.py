"""A word's edge punctuation and capitalization, which a distractor takes
on so that the two choices on screen look alike."""

from collections.abc import Callable


def letters_span(word: str) -> tuple[int, int]:
    """Where the word proper starts and ends: from its first alphanumeric
    character to its last. Empty for a word with none."""
    alphanumeric = [i for i, char in enumerate(word) if char.isalnum()]
    if not alphanumeric:
        return len(word), len(word)
    return alphanumeric[0], alphanumeric[-1] + 1


def strip_punctuation(word: str) -> str:
    start, end = letters_span(word)
    return word[start:end]


def case_of(word: str) -> Callable[[str], str]:
    """The transform that gives another word the capitalization of
    ``word``: all capitals, an initial capital, or neither. A one-letter
    word ("I", "A") counts as neither."""
    core = strip_punctuation(word)
    if len(core) > 1 and core.isupper():
        return str.upper
    if len(core) > 1 and core[0].isupper():
        return str.capitalize
    return str.lower


def copy_punctuation(word: str, distractor: str) -> str:
    """``distractor`` with the edge punctuation and capitalization of
    ``word``."""
    start, end = letters_span(word)
    return word[:start] + case_of(word)(distractor) + word[end:]
