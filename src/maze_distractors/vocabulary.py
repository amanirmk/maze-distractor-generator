"""The words a distractor may be, and the order to try them in at a given
position."""

import hashlib
import logging
import math
from collections.abc import Collection, Iterable, Sequence, Set
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Self

import wordfreq

from maze_distractors.punctuation import strip_punctuation

logger = logging.getLogger(__name__)

# Frequencies are natural-log occurrences per billion words, on which the
# least frequent word of the curated English list sits at 6.5 and "the"
# near 18.
_PER_BILLION = 1e9
# Log units of frequency within which a word counts as matching another.
_MATCHED = 1.0
# wordfreq's frequencies sit on a grid of 0.01 in log10, so two words are
# often exactly a factor of 1 or 10 apart, which the last digit of a float
# must not decide.
_TOLERANCE = 1e-9


def _frequencies(language: str, wordlist: str = "best") -> dict[str, float]:
    try:
        return wordfreq.get_frequency_dict(language, wordlist=wordlist)
    except LookupError as error:
        raise ValueError(
            f"wordfreq has no frequencies for the language {language!r}."
        ) from error


def hash_order(order_key: str, word: str) -> bytes:
    return hashlib.blake2b(
        f"{order_key}\0{word}".encode(), digest_size=8
    ).digest()


def _read_words(file: Path) -> set[str]:
    # utf-8-sig: a byte-order mark, which spreadsheets add, is not a letter
    # of the first word.
    return set(file.read_text(encoding="utf-8-sig").split())


def packaged_words(name: str) -> set[str]:
    data = resources.files("maze_distractors") / "data" / name
    return set(data.read_text(encoding="utf-8").split())


def _warn_of_left_out(
    include: Path, total: int, reason: str, left_out: Sequence[str]
) -> None:
    if left_out:
        logger.warning(
            "%d of the %d words of %s are left out, %s: %s",
            len(left_out),
            total,
            include,
            reason,
            ", ".join(left_out[:5]) + (", ..." if len(left_out) > 5 else ""),
        )


@dataclass(frozen=True)
class MatchRange:
    """The lengths a distractor for some words may have, and the range of
    those words' frequencies."""

    min_length: int
    max_length: int
    min_frequency: float
    max_frequency: float

    def distance(self, frequency: float) -> float:
        """Log units between ``frequency`` and the words' frequencies."""
        return max(
            self.min_frequency - frequency, frequency - self.max_frequency, 0.0
        )

    def tier(self, frequency: float) -> int:
        """0 for a frequency that matches the words, and one more for each
        further log unit away."""
        beyond = self.distance(frequency) - _MATCHED - _TOLERANCE
        return math.ceil(max(beyond, 0.0))


class Vocabulary:
    def __init__(self, words: Iterable[str], language: str = "en"):
        """``words`` that wordfreq has a ``language`` frequency for and that
        are lowercase letters only; the rest are dropped."""
        self.language = language
        # Words that cannot continue a noun phrase; English only, so empty
        # for any other language (scripts/build_noun_phrase_breakers.py).
        self.noun_phrase_breakers = (
            packaged_words("noun_phrase_breakers.txt")
            if language == "en"
            else frozenset()
        )
        frequencies = _frequencies(language)
        self._frequency = {
            word: math.log(frequencies[word] * _PER_BILLION)
            for word in set(words)
            if word in frequencies and word.isalpha() and word == word.lower()
        }
        self._rarest = min(self._frequency.values(), default=0.0)
        self.longest = max(map(len, self._frequency), default=0)

    @classmethod
    def load(
        cls,
        language: str = "en",
        include: Path | None = None,
        exclude: Sequence[Path] = (),
    ) -> Self:
        """The words of ``include`` -- by default the curated list for
        English, and for another language, with a warning, wordfreq's small
        list (words of at least one per million, screened for nothing) --
        minus those of every ``exclude`` file and, for English, the
        built-in exclusions and the words that are mainly proper nouns
        (scripts/build_proper_noun_list.py; rerun it after editing the
        curated list). Files hold one word per line."""
        if include is not None:
            words = _read_words(include)
        elif language == "en":
            words = packaged_words("curated_word_list.txt")
        else:
            words = set(_frequencies(language, wordlist="small"))
            logger.warning(
                "No curated word list for %r: falling back to every word "
                "of wordfreq's small list (%d words of at least one per "
                "million, screened for nothing). Give a curated list with "
                "--include.",
                language,
                len(words),
            )
        built_in: set[str] = set()
        if language == "en":
            built_in |= packaged_words("exclude.txt")
            built_in |= packaged_words("proper_nouns.txt")
        excluded = set(built_in)
        for file in exclude:
            excluded |= _read_words(file)
        vocabulary = cls(words - excluded, language)
        if include is not None:
            _warn_of_left_out(
                include,
                len(words),
                "being on the built-in exclusion or proper-noun lists",
                sorted(words & built_in),
            )
            _warn_of_left_out(
                include,
                len(words),
                "being capitalized, not letters only, or unknown to wordfreq",
                sorted((words - excluded) - vocabulary.words),
            )
        return vocabulary

    def __len__(self) -> int:
        return len(self._frequency)

    @property
    def words(self) -> Set[str]:
        return self._frequency.keys()

    def match_range(self, words: Sequence[str]) -> MatchRange:
        """Lengths within one letter of ``words``, and their frequencies.

        Both are clamped, since the very shortest, longest and most common
        words have too few equals to choose among: lengths always reach 4
        and start by 12, and a word more common than e^12 per billion
        (about 160 per million) has its range widened down to that. A word
        rarer than the whole vocabulary, or unknown to wordfreq, has no
        equals at all and is treated as the vocabulary's rarest.
        """
        cores = [strip_punctuation(word) for word in words]
        lengths = [len(core) for core in cores]
        frequencies = [
            wordfreq.zipf_frequency(core, self.language) * math.log(10)
            for core in cores
        ]
        return MatchRange(
            min_length=min(min(lengths) - 1, 12),
            max_length=max(max(lengths) + 1, 4),
            min_frequency=min(*frequencies, 12),
            max_frequency=max(*frequencies, self._rarest),
        )

    def candidates(
        self,
        words: Sequence[str],
        avoid: Collection[str],
        order_key: str,
        max_ratio: float = math.inf,
        *,
        extra_letters: int = 0,
        breaking_noun_phrase: bool = False,
    ) -> list[str]:
        """Every word whose length matches ``words`` or is up to
        ``extra_letters`` further off, whose frequency is within a factor
        of ``max_ratio`` of theirs, and that is not in ``avoid`` -- those
        matching in frequency first, then by whole log units beyond that.
        With ``breaking_noun_phrase``, only words that cannot continue a
        noun phrase.

        Within a distance the order is a hash of ``order_key`` and the word,
        not a shuffle: it is reproducible, and it does not change for the
        other words when one is added to an exclusion list.
        """
        band = self.match_range(words)
        shortest = band.min_length - extra_letters
        longest = band.max_length + extra_letters
        furthest = math.log(max_ratio) + _TOLERANCE
        ranked = [
            (band.tier(frequency), hash_order(order_key, word), word)
            for word, frequency in self._frequency.items()
            if shortest <= len(word) <= longest
            and band.distance(frequency) <= furthest
            and word not in avoid
            and (not breaking_noun_phrase or word in self.noun_phrase_breakers)
        ]
        return [word for _, _, word in sorted(ranked)]
