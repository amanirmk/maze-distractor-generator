import math

import pytest
import wordfreq

from maze_distractors.vocabulary import Vocabulary

WORDS = ["dog", "cat", "house", "table", "zebra", "the", "password", "run"]


def test_only_lowercase_words_with_a_frequency_are_kept():
    vocabulary = Vocabulary(["dog", "Dog", "don't", "x9", "qqzzxv", "dog"])
    assert len(vocabulary) == 1


def test_the_band_is_one_letter_either_side_and_the_words_frequencies():
    band = Vocabulary(WORDS).match_range(['"Tables,', "zebra"])
    assert (band.min_length, band.max_length) == (4, 7)
    assert band.min_frequency < band.max_frequency
    assert band.distance(band.max_frequency + 2.5) == 2.5
    assert band.distance(band.min_frequency) == 0


def test_within_one_log_unit_is_a_match_and_tiers_count_units_beyond():
    band = Vocabulary(WORDS).match_range(["zebra"])
    assert band.min_frequency == band.max_frequency
    tiers = [band.tier(band.min_frequency - d) for d in (0, 1, 1.01, 2, 2.5)]
    assert tiers == [0, 0, 1, 1, 2]


def test_a_very_common_word_is_matched_as_if_merely_common():
    # Nothing is as frequent as "the"; its distractors are matched to the
    # most common words there are enough of.
    band = Vocabulary(WORDS).match_range(["the"])
    assert band.min_frequency == 12
    assert band.max_frequency > 17


def test_a_word_rarer_than_the_vocabulary_is_matched_to_its_rarest():
    vocabulary = Vocabulary.load()
    rarest = min(vocabulary._frequency.values())
    for word in ("zorbled", "oscillated"):  # unknown to wordfreq; known, rare
        assert vocabulary.match_range([word]).max_frequency == rarest
        near = vocabulary.candidates([word], set(), "key", max_ratio=10)
        ratios = [math.exp(vocabulary._frequency[w] - rarest) for w in near]
        assert 9 < max(ratios) <= 10
    assert vocabulary.match_range(["garden"]).max_frequency > rarest


def test_candidates_match_length_and_skip_avoided_words():
    candidates = Vocabulary(WORDS).candidates(["horse"], {"table"}, "key")
    assert set(candidates) == {"house", "zebra"}


def test_words_matching_in_frequency_come_first():
    vocabulary = Vocabulary.load()
    band = vocabulary.match_range(["garden"])
    candidates = vocabulary.candidates(["garden"], set(), "key")
    tiers = [band.tier(vocabulary._frequency[word]) for word in candidates]
    assert tiers == sorted(tiers)
    assert tiers[0] == 0
    assert tiers[-1] > 0


def test_a_ratio_of_ten_keeps_words_within_an_order_of_magnitude():
    vocabulary = Vocabulary.load()
    garden = wordfreq.word_frequency("garden", "en")
    everything = vocabulary.candidates(["garden"], set(), "key")
    near = vocabulary.candidates(["garden"], set(), "key", max_ratio=10)
    ratios = [wordfreq.word_frequency(word, "en") / garden for word in near]
    # wordfreq rounds frequencies to three figures, hence the slack.
    assert 0.0995 < min(ratios) < 0.12
    assert 8 < max(ratios) < 10.05
    assert near == [word for word in everything if word in set(near)]
    assert 0 < len(near) < len(everything)


def test_a_ratio_of_one_keeps_the_words_of_equal_frequency():
    # Not an empty list: equal frequencies computed two ways differ in the
    # last digit of a float.
    vocabulary = Vocabulary.load()
    garden = vocabulary._frequency["garden"]
    equal = vocabulary.candidates(["garden"], {"garden"}, "key", max_ratio=1)
    assert equal
    assert all(
        vocabulary._frequency[word] == pytest.approx(garden) for word in equal
    )


def test_extra_letters_widen_the_lengths_let_in():
    vocabulary = Vocabulary(WORDS)
    assert vocabulary.longest == len("password")
    assert "house" not in vocabulary.candidates(["ox"], set(), "key")
    wider = vocabulary.candidates(["ox"], set(), "key", extra_letters=2)
    assert {"dog", "house", "table"} <= set(wider)
    assert "password" not in wider


def test_the_order_is_fixed_by_the_key():
    vocabulary = Vocabulary.load()
    first = vocabulary.candidates(["garden"], set(), "0\x001\x00verb")
    assert first == vocabulary.candidates(["garden"], set(), "0\x001\x00verb")
    assert first != vocabulary.candidates(["garden"], set(), "1\x001\x00verb")
    assert sorted(first) == sorted(
        vocabulary.candidates(["garden"], set(), "1\x001\x00verb")
    )


def test_excluding_a_word_leaves_the_order_of_the_rest(tmp_path):
    extra = tmp_path / "extra.txt"
    extra.write_text("\ufeffcastle\nbridge\n")  # with a byte-order mark
    before = Vocabulary.load().candidates(["garden"], set(), "key")
    after = Vocabulary.load(exclude=[extra]).candidates(
        ["garden"], set(), "key"
    )
    assert {"castle", "bridge"} <= set(before)
    assert after == [w for w in before if w not in {"castle", "bridge"}]


def test_the_default_list_is_curated_and_has_the_built_in_exclusions():
    vocabulary = Vocabulary.load()
    assert len(vocabulary) > 15_000
    assert "garden" in vocabulary._frequency
    assert "fuck" not in vocabulary._frequency


def test_words_that_are_mainly_proper_nouns_are_left_out(tmp_path):
    include = tmp_path / "include.txt"
    include.write_text("garden\njosh\noxford\n")
    assert Vocabulary.load(include=include).words == {"garden"}
    assert "josh" not in Vocabulary.load().words


def test_noun_phrase_breakers_are_finite_verbs_and_function_words():
    breakers = Vocabulary.load().noun_phrase_breakers
    # Finite verbs, pronouns, prepositions and conjunctions.
    assert {
        "acquires", "gave", "belong", "seems", "went", "brings", "became",
        "herself", "they", "whom", "ourselves",
        "onto", "unto", "toward",
        "unless", "nor", "although", "whereas",
    } <= breakers  # fmt: skip
    # Determiners and possessives: nothing continues a noun phrase after
    # "the the" either.
    assert {"the", "a", "his", "every", "each"} <= breakers
    # Participles, adverbs, nouns and adjectives; "no" is a noun too.
    assert not {
        "alluded", "running", "taken", "walked",
        "promptly", "quickly",
        "fifths", "table", "sunshine", "no",
        "effortless", "purple", "sturdy",
    } & breakers  # fmt: skip
    # Verbs that are as often nouns, whether WordNet knows it ("runs",
    # "despite") or not ("download", "edit").
    assert not {"runs", "despite", "download", "edit", "restart"} & breakers
    # "given" is the participle of "gave", and "leaned" one of two of
    # "lean"; "canning" and "hooded" are in no list of verb forms; "selfie"
    # and the noun "podcast" are newer than WordNet; "gived" is no word.
    assert not {
        "given", "leaned", "canning", "hooded", "selfie", "podcast", "gived"
    } & breakers  # fmt: skip


def test_candidates_can_be_limited_to_noun_phrase_breakers():
    vocabulary = Vocabulary.load()
    everything = vocabulary.candidates(["garden"], set(), "key")
    breaking = vocabulary.candidates(
        ["garden"], set(), "key", breaking_noun_phrase=True
    )
    assert breaking == [
        w for w in everything if w in vocabulary.noun_phrase_breakers
    ]
    assert 0 < len(breaking) < len(everything)


def test_an_include_file_replaces_the_curated_list(tmp_path):
    include = tmp_path / "include.txt"
    include.write_text("dog\ncat\nfuck\n")
    assert set(Vocabulary.load(include=include)._frequency) == {"dog", "cat"}


def test_another_language_uses_its_own_frequencies():
    french = Vocabulary(["chien", "maison", "dog"], language="fr")
    assert "chien" in french._frequency
    assert french.match_range(["maison"]).min_frequency > 0


def test_another_language_falls_back_to_the_small_list_with_a_warning(
    caplog,
):
    spanish = Vocabulary.load("es")
    # "la", "el" and "les" are in exclude.txt as English noise.
    assert {"la", "el", "les"} <= spanish.words
    assert not {"la", "el"} & Vocabulary.load().words
    assert 10_000 < len(spanish) < 40_000  # the small list, not the large
    assert min(spanish._frequency.values()) >= math.log(1e-6 * 1e9) - 0.01
    assert "No curated word list for 'es'" in caplog.text
    assert "--include" in caplog.text


def test_include_words_on_the_built_in_lists_are_named(tmp_path, caplog):
    include = tmp_path / "include.txt"
    include.write_text("garden\nfuck\njosh\n")
    assert Vocabulary.load(include=include).words == {"garden"}
    assert "2 of the 3 words" in caplog.text
    assert "built-in exclusion or proper-noun lists: fuck, josh" in caplog.text
    # A word the user excluded is not news.
    caplog.clear()
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("garden\n")
    Vocabulary.load(include=include, exclude=[exclude])
    assert "garden" not in caplog.text


def test_include_words_that_cannot_be_used_are_named(tmp_path, caplog):
    include = tmp_path / "include.txt"
    include.write_text("garden\nGarden\ndon't\nqqzzxv\n")
    assert Vocabulary.load(include=include).words == {"garden"}
    assert "3 of the 4 words" in caplog.text
    assert "Garden, don't, qqzzxv" in caplog.text
