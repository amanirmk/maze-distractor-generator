import math
import re

import pytest

from conftest import reference_surprisal
from maze_distractors.generation import (
    DETERMINERS,
    MISSING,
    PLACEHOLDER,
    Settings,
    generate,
)
from maze_distractors.items import Sentence
from maze_distractors.punctuation import case_of, strip_punctuation
from maze_distractors.vocabulary import Vocabulary

# Under the test model a token costs roughly 8-16 bits, so a 30-bit floor
# is met by some multi-token words at some positions and by none at others.
SETTINGS = Settings(min_delta=4.0, min_abs=30.0, max_repeat=0)


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load()


def sentence(tag, item, text, labels=None):
    words = tuple(text.split())
    labels = (
        tuple(labels.split()) if labels else tuple(map(str, range(len(words))))
    )
    return Sentence(tag, item, words, labels)


RELATIVE_CLAUSES = [
    sentence(
        "sub_rel",
        "3",
        "The cat who the dog scared hid in a box.",
        "pre_1 pre_2 who art noun verb main_verb post_1 post_2 post_3",
    ),
    sentence(
        "obj_rel",
        "3",
        "The dog who scared the cat sniffed around the couch.",
        "pre_1 pre_2 who verb art noun main_verb post_1 post_2 post_3",
    ),
]
PLAIN = [
    sentence("a", "1", "The dog chased the cat around the house."),
    sentence("a", "2", "A bird sang in the tree, LOUDLY."),
]


def bare(distractor):
    return strip_punctuation(distractor).lower()


def scored_surprisal(scorer, context, distractor, often_capitalized):
    """A distractor counts for the lowest of its surprisal as shown, in
    lower case and, for a word often capitalized, capitalized, without edge
    punctuation any way."""
    shown = strip_punctuation(distractor)
    forms = [shown, shown.lower()]
    if shown.lower() in often_capitalized:
        forms.append(shown.capitalize())
    return min(reference_surprisal(scorer, context, form) for form in forms)


def exhaustive_choice(
    scorer, vocabulary, settings, item, label, target_words, avoid
):
    """The specified choice, made without any shortcut: exact surprisal of
    every candidate at every target word, shown in that word's
    capitalization and counted for the lowest of that, lower case and, for a
    word often capitalized, capitalized, in
    candidate order, the length match widened a letter at a time until
    there are enough candidates."""
    thresholds = [
        max(
            settings.min_abs,
            reference_surprisal(scorer, context, word) + settings.min_delta,
        )
        for context, word in target_words
    ]
    best, best_shortfall = None, math.inf
    after_determiner = any(
        strip_punctuation(context.split()[-1]).lower() in DETERMINERS
        for context, _ in target_words
    )
    words = [word for _, word in target_words]
    reach = max(vocabulary.longest, vocabulary.match_range(words).min_length)
    for extra_letters in range(reach + 1):
        candidates = vocabulary.candidates(
            words,
            avoid,
            f"{settings.seed}\0{item}\0{label}",
            settings.max_frequency_ratio,
            extra_letters=extra_letters,
        )
        if after_determiner:
            candidates = [
                c for c in candidates if c in vocabulary.noun_phrase_breakers
            ]
        if len(candidates) >= settings.min_candidates:
            break
    for candidate in candidates:
        shortfall = max(
            threshold
            - scored_surprisal(
                scorer,
                context,
                case_of(word)(candidate),
                vocabulary.often_capitalized,
            )
            for threshold, (context, word) in zip(
                thresholds, target_words, strict=True
            )
        )
        if shortfall <= 0:
            return candidate
        if shortfall < best_shortfall:
            best, best_shortfall = candidate, shortfall
    return best


def _compare_with_exhaustive_search(scorer, vocabulary, settings, sentences):
    """Generate for one item and check each label's choice against
    ``exhaustive_choice``. Returns the positions."""
    (item,) = {s.item for s in sentences}
    _, positions = generate(sentences, scorer, vocabulary, settings)
    target_words: dict[str, list[tuple[str, str]]] = {}
    for s in sentences:
        for i in range(1, len(s.words)):
            target_words.setdefault(s.labels[i], []).append(
                (" ".join(s.words[:i]), s.words[i])
            )
    used: set[str] = set()
    # Whole words and their letter runs: "cat" of "cat's" is a word the
    # reader has just seen.
    item_words = {
        part
        for s in sentences
        for word in s.words
        for part in [bare(word), *re.findall(r"[^\W\d_]+", word.lower())]
    }
    for label, label_words in target_words.items():
        expected = exhaustive_choice(
            scorer, vocabulary, settings, item, label, label_words,
            used | item_words,
        )  # fmt: skip
        used.add(expected)
        chosen = {
            bare(p.distractor)
            for p in positions
            if p.sentence.labels[p.index] == label
        }
        assert chosen == {expected}, label
    return positions


def _reference_vocabulary(vocabulary, extra=()):
    # A few hundred words, since the reference scores every one of them.
    breakers = vocabulary.noun_phrase_breakers & vocabulary.words
    return Vocabulary(
        sorted(vocabulary.words)[::60] + sorted(breakers)[::8] + list(extra)
    )


def test_the_shortcuts_choose_what_an_exhaustive_search_chooses(
    scorer, vocabulary, caplog
):
    # A floor high enough that some positions find none, so the fallback
    # is compared as well as the first-to-pass rule; and a minimum some
    # positions fall short of, so the widened length match is compared too.
    settings = Settings(
        min_delta=4.0, min_abs=58.0, max_repeat=0, min_candidates=20
    )
    positions = _compare_with_exhaustive_search(
        scorer, _reference_vocabulary(vocabulary), settings, RELATIVE_CLAUSES
    )
    # Both outcomes occur, so both branches of the search were compared.
    assert {p.met for p in positions} == {True, False}
    assert "further off" in caplog.text
    # A position that fell short is logged, as the README says.
    assert "bits short" in caplog.text


def test_the_exhaustive_search_agrees_on_capitals_and_word_parts(
    scorer, vocabulary
):
    # Capitalized targets are held to both forms, and the parts of "cat's"
    # and "well-known" are in the vocabulary but may not be chosen.
    sentences = [
        sentence("a", "7", "Yesterday the cat's owner met Boston's well-known MAYOR."),
        sentence("b", "7", "Yesterday the cat's owner met boston's well-known mayor."),
    ]  # fmt: skip
    settings = Settings(min_delta=4.0, min_abs=40.0, max_repeat=0)
    reference = _reference_vocabulary(vocabulary, ["cat", "well", "known"])
    positions = _compare_with_exhaustive_search(
        scorer, reference, settings, sentences
    )
    chosen = {bare(p.distractor) for p in positions}
    assert not chosen & {"cat", "well", "known"}


def test_the_report_holds_exact_surprisals_and_each_words_own_threshold(
    scorer, vocabulary
):
    _, positions = generate(RELATIVE_CLAUSES, scorer, vocabulary, SETTINGS)
    for p in positions:
        context = " ".join(p.sentence.words[: p.index])
        real = reference_surprisal(scorer, context, p.sentence.words[p.index])
        assert p.threshold == pytest.approx(max(30.0, real + 4.0), abs=1e-3)
        assert p.surprisal == pytest.approx(
            scored_surprisal(
                scorer, context, p.distractor, vocabulary.often_capitalized
            ),
            abs=1e-3,
        )


def test_after_a_determiner_only_a_word_that_ends_the_noun_phrase_is_tried(
    scorer, vocabulary
):
    item = [
        sentence(
            "a", "1", 'The dog chased "the cat around \u201cthe bird, his.'
        )
    ]
    (distracted,) = generate(item, scorer, vocabulary, SETTINGS)[0]
    words = dict(
        zip(
            distracted.sentence.words,
            map(bare, distracted.distractors),
            strict=True,
        )
    )
    assert words["dog"] in vocabulary.noun_phrase_breakers
    assert words["cat"] in vocabulary.noun_phrase_breakers  # after '"the'
    assert words["bird,"] in vocabulary.noun_phrase_breakers  # a curly quote
    # The rest follow a noun or a verb: no restriction, and among 19,000
    # words the first to pass is not one of the 1,100 that do.
    free = {words["chased"], words["around"], words["his."]}
    assert not free <= vocabulary.noun_phrase_breakers


def test_the_noun_phrase_rule_can_be_turned_off_and_skips_other_languages(
    scorer, vocabulary
):
    item = [sentence("a", "1", "The dog barked.")]
    off = Settings(min_delta=4.0, min_abs=30.0, break_noun_phrases=False)
    picks = set()
    for seed in range(8):
        settings = Settings(min_delta=4.0, min_abs=30.0, seed=seed)
        (on,) = generate(item, scorer, vocabulary, settings)[0]
        assert bare(on.distractors[1]) in vocabulary.noun_phrase_breakers
        (without,) = generate(
            item, scorer, vocabulary, Settings(**{**vars(off), "seed": seed})
        )[0]
        picks.add(bare(without.distractors[1]))
    assert not picks <= vocabulary.noun_phrase_breakers
    assert (
        Vocabulary(["chien"], language="fr").noun_phrase_breakers == frozenset()
    )


def test_a_language_without_breakers_warns_that_the_rule_is_off(scorer, caplog):
    item = [sentence("a", "1", "Le chien a mange.")]
    french = Vocabulary(["chat", "maison", "table"], language="fr")
    generate(item, scorer, french, SETTINGS)
    assert (
        "no list of noun-phrase breakers for the language 'fr'" in caplog.text
    )
    caplog.clear()
    off = Settings(**{**vars(SETTINGS), "break_noun_phrases": False})
    generate(item, scorer, french, off)
    assert "noun-phrase breakers" not in caplog.text


def test_a_label_gets_one_distractor_wherever_it_appears(scorer, vocabulary):
    distracted, _ = generate(RELATIVE_CLAUSES, scorer, vocabulary, SETTINGS)
    by_label = [
        dict(zip(d.sentence.labels, map(bare, d.distractors), strict=True))
        for d in distracted
    ]
    assert by_label[0]["verb"] == by_label[1]["verb"]
    assert by_label[0]["noun"] == by_label[1]["noun"]
    assert by_label[0] == by_label[1]


def test_distractors_follow_the_words_punctuation_and_capitals(
    scorer, vocabulary
):
    distracted, _ = generate(PLAIN, scorer, vocabulary, SETTINGS)
    first, second = (d.distractors for d in distracted)
    assert first[0] == second[0] == PLACEHOLDER
    assert first[-1].endswith(".")
    assert first[-1][:-1].islower()
    assert second[-2].endswith(",")
    assert second[-1] != MISSING
    assert second[-1].endswith(".")
    assert second[-1] == second[-1].upper()
    assert all(len(d.distractors) == len(d.sentence.words) for d in distracted)


def test_a_capitalized_distractor_counts_for_the_lower_of_both_forms(
    scorer, vocabulary
):
    # One label, one context, three capitalizations: one distractor, and at
    # each target word the lower surprisal of the form shown and lower case.
    sentences = [
        sentence("a", "1", "We saw boston today."),
        sentence("b", "1", "We saw Boston today."),
        sentence("c", "1", "We saw BOSTON today."),
    ]
    _, positions = generate(sentences, scorer, vocabulary, SETTINGS)
    shown = [p.distractor for p in positions if p.index == 2]
    assert len({d.lower() for d in shown}) == 1
    assert [d == d.lower() for d in shown] == [True, False, False]
    assert shown[1] == shown[0].capitalize()
    assert shown[2] == shown[0].upper()
    surprisals = [p.surprisal for p in positions if p.index == 2]
    as_shown = [reference_surprisal(scorer, "We saw", d) for d in shown]
    assert len(set(as_shown)) == 3
    for form, surprisal in zip(as_shown, surprisals, strict=True):
        assert surprisal == pytest.approx(min(form, as_shown[0]), abs=1e-3)
    # The search was held to the lowest of them all.
    assert min(surprisals) == pytest.approx(min(as_shown), abs=1e-3)


def test_sentences_of_an_item_need_not_be_adjacent(scorer, vocabulary):
    sentences = [
        sentence("a", "1", "The dog chased the cat."),
        sentence("b", "2", "A bird sang in the tree."),
        sentence("c", "1", "The dog chased the mouse."),
    ]
    distracted, positions = generate(sentences, scorer, vocabulary, SETTINGS)
    assert [d.sentence.tag for d in distracted] == ["a", "b", "c"]
    assert [p.sentence.tag for p in positions] == ["a"] * 4 + ["b"] * 5 + [
        "c"
    ] * 4
    assert distracted[0].distractors[:4] == distracted[2].distractors[:4]
    assert distracted[0].distractors[:4] != distracted[1].distractors[:4]


def test_when_no_length_matches_the_nearest_length_is_tried(scorer, caplog):
    long_words = Vocabulary(["establishment", "international"])
    settings = Settings(min_abs=0.0, max_frequency_ratio=math.inf)
    (distracted,) = generate(
        [sentence("a", "1", "Maybe birds sleep.")], scorer, long_words, settings
    )[0]
    assert {bare(d) for d in distracted.distractors[1:]} == set(
        long_words.words
    )
    assert "0 candidate(s) at the matching lengths" in caplog.text


def test_too_few_candidates_let_in_lengths_further_off_but_after(
    scorer, caplog
):
    # Any word meets a threshold this low, so the first candidate is chosen.
    anything = {"min_delta": -1000.0, "min_abs": 0.0}
    item = [sentence("a", "1", "Maybe dog.")]
    words = Vocabulary(["cat", "horse"])

    def tried_first(min_candidates):
        settings = Settings(
            **anything,
            max_frequency_ratio=math.inf,
            min_candidates=min_candidates,
        )
        (distracted,) = generate(item, scorer, words, settings)[0]
        return bare(distracted.distractors[1])

    assert tried_first(1) == "cat"
    assert "further off" not in caplog.text
    # "horse" is let in, but after "cat", which is closer in length.
    assert tried_first(2) == "cat"
    assert "1 candidate(s) at the matching lengths" in caplog.text
    # With nothing further off to let in, there is nothing to warn of.
    caplog.clear()
    generate(
        item,
        scorer,
        Vocabulary(["cat", "sun"]),
        Settings(**anything, min_candidates=10),
    )
    assert "further off" not in caplog.text


def test_lengths_are_widened_downward_as_far_as_the_vocabulary_needs(scorer):
    short_words = Vocabulary(["cat", "dog", "sun", "sat", "run"])
    settings = Settings(min_abs=0.0, max_frequency_ratio=math.inf)
    (distracted,) = generate(
        [sentence("a", "1", "Maybe establishment sleeps.")],
        scorer,
        short_words,
        settings,
    )[0]
    assert MISSING not in distracted.distractors
    assert {bare(d) for d in distracted.distractors[1:]} <= set(
        short_words.words
    )


def test_no_distractor_repeats_within_an_item_or_is_the_real_word(
    scorer, vocabulary
):
    distracted, _ = generate(PLAIN, scorer, vocabulary, SETTINGS)
    for d in distracted:
        chosen = [bare(word) for word in d.distractors[1:]]
        assert len(set(chosen)) == len(chosen)
        assert not set(chosen) & {bare(word) for word in d.sentence.words}


def test_the_parts_of_a_hyphenated_or_possessive_word_are_not_distractors(
    scorer,
):
    parts = Vocabulary(["cat", "well", "known", "owner"])
    item = [sentence("a", "1", "The cat's owner was well-known.")]
    distracted, _ = generate(item, scorer, parts, Settings(min_abs=0.0))
    assert set(distracted[0].distractors[1:]) == {MISSING}


def test_no_real_word_of_the_item_is_a_distractor_anywhere_in_it(scorer):
    # "Stone," is the only word there is: never its own distractor, and
    # not "fell."'s either, since the reader has just seen it.
    only = Vocabulary(["stone"])
    item = [sentence("a", "1", "Maybe Stone, fell.")]
    distracted, _ = generate(item, scorer, only, Settings(min_abs=0.0))
    assert distracted[0].distractors == (PLACEHOLDER, MISSING, MISSING)
    # Nor a word of the item's other sentence.
    two = Vocabulary(["stone", "brick"])
    item = [
        sentence("a", "1", "Maybe brick fell."),
        sentence("b", "1", "Maybe Stone, fell."),
    ]
    distracted, _ = generate(item, scorer, two, Settings(min_abs=0.0))
    assert {d for x in distracted for d in x.distractors} == {
        PLACEHOLDER,
        MISSING,
    }


def test_max_repeat_limits_uses_across_items(scorer):
    # Seven five-letter words for the ten target_words of two items.
    small = Vocabulary(
        ["table", "chair", "stone", "plant", "river", "cloud", "horse"]
    )
    items = [
        sentence("a", "1", "Maybe those birds never sleep there."),
        sentence("a", "2", "Often small child plays games daily."),
    ]

    def uses(max_repeat):
        distracted, _ = generate(
            items, scorer, small, Settings(max_repeat=max_repeat)
        )
        chosen = [bare(w) for d in distracted for w in d.distractors[1:]]
        return {word: chosen.count(word) for word in set(chosen)}

    assert max(uses(0).values()) == 2
    once = uses(1)
    assert once.pop(bare(MISSING)) == 3
    assert list(once.values()) == [1] * 7


def test_a_place_with_no_candidate_gets_the_placeholder(scorer, caplog):
    empty = Vocabulary([])
    distracted, positions = generate(PLAIN, scorer, empty, SETTINGS)
    assert distracted[0].distractors[0] == PLACEHOLDER
    assert set(distracted[0].distractors[1:]) == {MISSING}
    assert math.isnan(positions[0].surprisal)
    assert not positions[0].met
    assert "no word to try" in caplog.text


def test_generation_is_reproducible_and_the_seed_changes_it(scorer, vocabulary):
    def run(seed):
        settings = Settings(min_delta=4.0, min_abs=20.0, seed=seed)
        return [
            d.distractors
            for d in generate(PLAIN, scorer, vocabulary, settings)[0]
        ]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_the_frequency_cap_is_kept_even_when_no_word_meets_the_threshold(
    scorer, vocabulary
):
    item = [sentence("a", "1", "The dog barked.")]

    def distances(ratio):
        settings = Settings(min_abs=1000.0, max_frequency_ratio=ratio)
        (distracted,) = generate(item, scorer, vocabulary, settings)[0]
        return [
            vocabulary.match_range([word]).distance(
                vocabulary._frequency[bare(d)]
            )
            for word, d in zip(
                distracted.sentence.words[1:],
                distracted.distractors[1:],
                strict=True,
            )
        ]

    assert max(distances(10.0)) <= math.log2(10) + 1e-9
    assert max(distances(math.inf)) > math.log2(10)
