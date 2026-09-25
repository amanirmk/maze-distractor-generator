"""Choose a distractor for every labelled position of every item."""

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Collection, Sequence, Set
from dataclasses import dataclass

from maze_distractors.items import Sentence
from maze_distractors.punctuation import (
    case_of,
    copy_punctuation,
    strip_punctuation,
)
from maze_distractors.surprisal import Context, ScoredWord, Scorer
from maze_distractors.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Shown beside a sentence's first word, which has no distractor by design.
PLACEHOLDER = "x-x-x"
# Shown where no word of the vocabulary could be tried: plainly not a
# distractor, so that it cannot pass unnoticed into an experiment.
MISSING = "NO-DISTRACTOR"
# Words that are always determiners, so that a noun phrase is open after
# them. "that", "this" and "her" are as often something else.
DETERMINERS = frozenset(
    ["the", "a", "an", "my", "your", "his", "its", "our", "their", "every"]
)


@dataclass(frozen=True)
class Settings:
    """A distractor's surprisal threshold is ``min_delta`` bits above the
    threshold word's, and never below ``min_abs``."""

    min_delta: float = 10.0
    min_abs: float = 25.0
    max_repeat: int = 1  # uses of one distractor across all items; 0 = any
    # Times rarer or more common than the threshold words a distractor may be;
    # math.inf to meet the threshold whatever the frequency.
    max_frequency_ratio: float = 10.0
    # Words the length window must let in before it stops widening, so that
    # a position with only a few words close in length still has a choice.
    # The closest lengths are still tried first.
    min_candidates: int = 10
    # After a determiner, try only words that cannot continue the noun
    # phrase: there a noun, adjective, participle or adverb could go on to
    # make a sentence however surprising it is ("The alluded...").
    break_noun_phrases: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        # nan compares false with everything, so it would pass every
        # threshold check and then break the run record.
        for name, value in vars(self).items():
            if isinstance(value, float) and math.isnan(value):
                raise ValueError(f"The setting {name} is nan.")


@dataclass(frozen=True)
class SentenceDistractors:
    sentence: Sentence
    distractors: tuple[str, ...]  # one per word, PLACEHOLDER first


@dataclass(frozen=True)
class ChosenDistractor:
    """The distractor chosen for one threshold word, and how it fared."""

    sentence: Sentence
    index: int
    distractor: str
    threshold: float
    # Exact, in bits; the lower of as shown and in lower case; nan for
    # MISSING.
    surprisal: float

    @property
    def met(self) -> bool:
        return self.surprisal >= self.threshold


@dataclass(frozen=True)
class _TargetWord:
    """A word of the input that needs a distractor: where it stands, its
    surprisal, and the threshold that surprisal sets."""

    sentence_number: int
    index: int
    word: str
    scored: ScoredWord
    threshold: float
    follows_determiner: bool


@dataclass(frozen=True)
class _ContextVariant:
    """One of the settings a candidate is scored in: after a context, in
    one capitalization, against the highest threshold set there. A
    candidate has to clear every variant of its label."""

    context: Context
    recase: Callable[[str], str]
    threshold: float


def _target_words_by_label(
    sentences: Sequence[Sentence], scorer: Scorer, settings: Settings
) -> dict[str, list[_TargetWord]]:
    """Every word after a sentence's first, under its label; labels in
    order of first appearance, which fixes the order they are settled in."""
    target_words: dict[str, list[_TargetWord]] = {}
    for number, sentence in enumerate(sentences):
        for index, scored in enumerate(
            scorer.score_words(sentence.words), start=1
        ):
            threshold = max(
                settings.min_abs, scored.surprisal + settings.min_delta
            )
            previous = strip_punctuation(sentence.words[index - 1]).lower()
            target_words.setdefault(sentence.labels[index], []).append(
                _TargetWord(
                    number,
                    index,
                    sentence.words[index],
                    scored,
                    threshold,
                    follows_determiner=previous in DETERMINERS,
                )
            )
    return target_words


def _candidates(
    item: str,
    label: str,
    target_words: Sequence[_TargetWord],
    vocabulary: Vocabulary,
    settings: Settings,
    avoid: Collection[str],
) -> list[str]:
    """The words to try at ``target_words``, in order. When fewer than
    ``settings.min_candidates`` pass the filters at the matching lengths --
    length, frequency, the words to avoid and the noun-phrase rule together
    -- lengths a letter further off are let in until enough do: a poorer
    match on screen is better than no distractor, or than one forced by
    having nothing else to try. The words let in are tried after those
    closer in length."""
    candidates: list[str] = []
    matching = 0  # how many passed at the matching lengths
    widened_to = 0  # letters further off than those, where the last came in
    words = [target_word.word for target_word in target_words]
    # Far enough to reach the vocabulary's longest word upward and its
    # shortest downward.
    reach = max(vocabulary.longest, vocabulary.match_range(words).min_length)
    for extra_letters in range(reach + 1):
        wider = vocabulary.candidates(
            words,
            avoid=avoid,
            order_key=f"{settings.seed}\0{item}\0{label}",
            max_ratio=settings.max_frequency_ratio,
            extra_letters=extra_letters,
            breaking_noun_phrase=(
                settings.break_noun_phrases
                and bool(vocabulary.noun_phrase_breakers)
                and any(
                    target_word.follows_determiner
                    for target_word in target_words
                )
            ),
        )
        if extra_letters == 0:
            matching = len(wider)
        elif len(wider) > len(candidates):
            widened_to = extra_letters
        candidates = wider
        if len(candidates) >= settings.min_candidates:
            break
    # Only when the words let in are there to be tried: a vocabulary with
    # nothing further off is not worth a warning at every position.
    if widened_to:
        logger.warning(
            "Item %s, label %s: %d candidate(s) at the matching lengths "
            "(length, frequency, repeats and the noun-phrase rule together), "
            "fewer than %d; trying lengths up to %d letter(s) further off.",
            item,
            label,
            matching,
            settings.min_candidates,
            widened_to,
        )
    return candidates


# A context's tokens and a capitalization.
type _VariantKey = tuple[tuple[int, ...], Callable[[str], str]]


def _variant_keys(
    target_word: _TargetWord, *, often_capitalized: bool = False
) -> list[_VariantKey]:
    """A distractor shown capitalized is scored as shown and in lower case,
    and has to meet the threshold both ways: the model can take a capitalized
    word for a name, which a reader who knows the word does not. A word
    that is often capitalized is scored capitalized too, whichever way it is
    shown: a reader can take "josh" for the name, which the model, scoring
    the lower case, does not."""
    tokens = target_word.scored.context.token_ids
    shown = (tokens, case_of(target_word.word))
    keys = [shown, (tokens, str.lower)]
    if often_capitalized:
        keys.append((tokens, str.capitalize))
    return list(dict.fromkeys(keys))


def variants_for(
    target_words: Sequence[_TargetWord], *, often_capitalized: bool = False
) -> dict[_VariantKey, _ContextVariant]:
    """Sentences of an item often share their opening words. A candidate is
    scored once for each distinct context and capitalization, against the
    highest threshold set there."""
    variants: dict[_VariantKey, _ContextVariant] = {}
    for target_word in target_words:
        for key in _variant_keys(
            target_word, often_capitalized=often_capitalized
        ):
            threshold = max(
                target_word.threshold,
                variants[key].threshold if key in variants else -math.inf,
            )
            variants[key] = _ContextVariant(
                target_word.scored.context, key[1], threshold
            )
    return variants


def score_in_variants(
    scorer: Scorer,
    variants: Sequence[_ContextVariant],
    candidates: Sequence[str],
    *,
    exactly: bool,
) -> list[list[float]]:
    """``result[candidate][variant]``, each candidate capitalized as its
    variant shows it. Exact throughout, or exact only below each variant's
    threshold (see ``Scorer.score_candidates``)."""
    result = [[math.nan] * len(variants) for _ in candidates]
    for recase in dict.fromkeys(variant.recase for variant in variants):
        columns = [
            j for j, variant in enumerate(variants) if variant.recase is recase
        ]
        contexts = [variants[j].context for j in columns]
        recased = [recase(candidate) for candidate in candidates]
        if exactly:
            scores = scorer.score_exactly(contexts, recased)
        else:
            bounds = [variants[j].threshold for j in columns]
            scores = scorer.score_candidates(contexts, recased, bounds)
        for k, row in enumerate(scores):
            for j, surprisal in zip(columns, row, strict=True):
                result[k][j] = surprisal
    return result


def _score_chunk(
    scorer: Scorer,
    variants_of: Callable[[str], Sequence[_ContextVariant]],
    chunk: Sequence[str],
) -> dict[str, list[float]]:
    """Each word's surprisal in its variants, exact below their thresholds;
    words held to the same variants are scored together."""
    groups: dict[int, tuple[Sequence[_ContextVariant], list[str]]] = {}
    for word in chunk:
        variants = variants_of(word)
        groups.setdefault(id(variants), (variants, []))[1].append(word)
    scores = {}
    for variants, words in groups.values():
        scored = score_in_variants(scorer, variants, words, exactly=False)
        scores.update(zip(words, scored, strict=True))
    return scores


def _search(
    scorer: Scorer,
    variants_of: Callable[[str], Sequence[_ContextVariant]],
    candidates: Sequence[str],
) -> tuple[str | None, list[float]]:
    """The first candidate to meet the threshold in every one of its
    variants (``variants_of`` it), with its exact surprisal in each. When
    none does, the one whose worst shortfall is smallest, the earliest of
    equals; None and an empty list without candidates.

    Candidates are scored in chunks that double, since a position is either
    settled within a few words or needs most of the vocabulary.
    """

    def shortfall(word: str, surprisals: Sequence[float]) -> float:
        return max(
            variant.threshold - s
            for variant, s in zip(variants_of(word), surprisals, strict=True)
        )

    def score_exactly(word: str) -> list[float]:
        return score_in_variants(
            scorer, variants_of(word), [word], exactly=True
        )[0]

    best, best_shortfall = None, math.inf
    start, size = 0, 16
    while start < len(candidates):
        chunk = candidates[start : start + size]
        scores = _score_chunk(scorer, variants_of, chunk)
        for word in chunk:
            short = shortfall(word, scores[word])
            if short <= _NEAR_MISS:
                # Settled on the exact values that get reported, which can
                # differ in the last digits from a bound or a batch -- in
                # either direction, so a near miss is rechecked too.
                confirmed = score_exactly(word)
                short = shortfall(word, confirmed)
                if short <= 0:
                    return word, confirmed
            if short < best_shortfall:
                best, best_shortfall = word, short
        start, size = start + size, size * 2
    if best is None:
        return None, []
    return best, score_exactly(best)


# Bits. A shortfall this small in a padded batch may be float noise.
_NEAR_MISS = 1e-3


def _choose(
    scorer: Scorer,
    target_words: Sequence[_TargetWord],
    candidates: Sequence[str],
    often_capitalized: Set[str],
) -> tuple[str | None, list[float]]:
    """The distractor for ``target_words`` and its exact surprisal at each, or
    None and nans when there is no candidate. A candidate in
    ``often_capitalized`` is held to its threshold capitalized too."""
    variants = {
        listed: variants_for(target_words, often_capitalized=listed)
        for listed in (False, True)
    }
    as_lists = {listed: list(v.values()) for listed, v in variants.items()}
    distractor, surprisals = _search(
        scorer,
        lambda word: as_lists[word in often_capitalized],
        candidates,
    )
    if distractor is None:
        return None, [math.nan] * len(target_words)
    listed = distractor in often_capitalized
    by_variant = dict(zip(variants[listed], surprisals, strict=True))
    return distractor, [
        min(
            by_variant[key]
            for key in _variant_keys(target_word, often_capitalized=listed)
        )
        for target_word in target_words
    ]


def _distract_item(
    item: str,
    sentences: Sequence[Sentence],
    scorer: Scorer,
    vocabulary: Vocabulary,
    settings: Settings,
    uses: Counter[str],
) -> list[list[ChosenDistractor]]:
    """Each sentence's chosen, in word order. ``uses`` counts the words
    chosen so far across items, and is updated."""
    chosen: dict[tuple[int, int], ChosenDistractor] = {}
    used_here: set[str] = set()
    # No word of the item, wherever it stands: "the" beside "the", or a
    # word the reader has just seen, is no test of reading. That includes
    # the parts of "cat's" and "well-known".
    item_words = {
        part
        for s in sentences
        for word in s.words
        for part in [
            strip_punctuation(word).lower(),
            *re.findall(r"[^\W\d_]+", word.lower()),
        ]
    }
    for label, target_words in _target_words_by_label(
        sentences, scorer, settings
    ).items():
        worn_out = {w for w, n in uses.items() if 0 < settings.max_repeat <= n}
        candidates = _candidates(
            item,
            label,
            target_words,
            vocabulary,
            settings,
            avoid=used_here | worn_out | item_words,
        )
        distractor, surprisals = _choose(
            scorer, target_words, candidates, vocabulary.often_capitalized
        )
        if distractor is not None:
            used_here.add(distractor)
            uses[distractor] += 1
        for target_word, surprisal in zip(
            target_words, surprisals, strict=True
        ):
            chosen[target_word.sentence_number, target_word.index] = (
                ChosenDistractor(
                    sentence=sentences[target_word.sentence_number],
                    index=target_word.index,
                    distractor=(
                        copy_punctuation(target_word.word, distractor)
                        if distractor is not None
                        else MISSING
                    ),
                    threshold=target_word.threshold,
                    surprisal=surprisal,
                )
            )
        shortfall = max(
            p.threshold - s
            for p, s in zip(target_words, surprisals, strict=True)
        )
        if distractor is None:
            logger.warning("Item %s, label %s: no word to try.", item, label)
        elif shortfall > 0:
            logger.warning(
                "Item %s, label %s: no word met the threshold; %r is %.1f bits "
                "short.",
                item,
                label,
                distractor,
                shortfall,
            )
    return [
        [chosen[number, index] for index in range(1, len(sentence.words))]
        for number, sentence in enumerate(sentences)
    ]


def generate(
    sentences: Sequence[Sentence],
    scorer: Scorer,
    vocabulary: Vocabulary,
    settings: Settings,
) -> tuple[list[SentenceDistractors], list[ChosenDistractor]]:
    """Distractors for every sentence, and a report of every position, both
    in the order of ``sentences``. Sentences with the same ``item`` share
    distractors; items are settled in order of first appearance."""
    if settings.break_noun_phrases and not vocabulary.noun_phrase_breakers:
        logger.warning(
            "There is no list of noun-phrase breakers for the language %r "
            "(English only), so nothing limits the words tried after a "
            "determiner.",
            vocabulary.language,
        )
    by_item: dict[str, list[int]] = {}
    for number, sentence in enumerate(sentences):
        by_item.setdefault(sentence.item, []).append(number)
    uses: Counter[str] = Counter()
    per_sentence: dict[int, list[ChosenDistractor]] = {}
    for item, numbers in by_item.items():
        logger.info("Item %s", item)
        of_item = [sentences[number] for number in numbers]
        per_sentence.update(
            zip(
                numbers,
                _distract_item(
                    item, of_item, scorer, vocabulary, settings, uses
                ),
                strict=True,
            )
        )
    distracted = [
        SentenceDistractors(
            sentence,
            (PLACEHOLDER, *(p.distractor for p in per_sentence[number])),
        )
        for number, sentence in enumerate(sentences)
    ]
    chosen = [
        p for number in range(len(sentences)) for p in per_sentence[number]
    ]
    return distracted, chosen
