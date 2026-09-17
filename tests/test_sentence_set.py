import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_set import Label, Sentence, Sentence_Set
from limit_repeats import Repeatcounter

PARAMS = {"min_abs": 25, "min_delta": 10, "num_to_test": 100}


def make_label(word="cat", prefix="The", surprisal=8.0):
    lab = Label("1", 1)
    lab.add_sentence(word, prefix, surprisal)
    return lab


def mock_dict(words):
    d = MagicMock()
    d.get_potential_distractors.return_value = list(words)
    return d


def threshold_func(words):
    return (2, 10, 1, 20)


def make_sentence_set():
    """Build a minimal Sentence_Set with surprisals pre-set (no model needed)."""
    ss = Sentence_Set("1")
    sent = Sentence(["The", "cat", "sat"], [0, 1, 2], "1", "test")
    sent.surprisal = {1: 8.0, 2: 8.0}
    ss.add(sent)
    ss.make_labels()
    return ss


# --- Label.choose_top_n_distractors ---

class TestChooseTopN:
    def test_returns_n_good_candidates(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast", "jump", "red"]), threshold_func, PARAMS, [], "1"
            )
        assert len(result) == 3

    def test_candidates_are_from_pool(self):
        lab = make_label()
        pool = ["dog", "run", "fast", "jump", "red"]
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(3, None, mock_dict(pool), threshold_func, PARAMS, [], "1")
        assert all(w in pool for w in result)

    def test_fills_with_fallbacks_if_fewer_good(self):
        lab = make_label()
        # target = max(min_abs=25, surprisal=8+min_delta=10) = 25
        # "dog" returns 30 (>= 25 threshold), others return 5 (< 25)
        def fake_surprisal(backend, prefix, word):
            return 30.0 if word == "dog" else 5.0
        with patch("sentence_set.get_surprisal", side_effect=fake_surprisal):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1"
            )
        assert len(result) == 3
        assert result[0] == "dog"  # good candidate first

    def test_respects_banned(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                2, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, ["dog"], "1"
            )
        assert "dog" not in result

    def test_sets_self_distractor_to_first(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1"
            )
        assert lab.distractor == result[0]

    def test_returns_empty_and_sets_fallback_when_no_candidates(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict([]), threshold_func, PARAMS, [], "1"
            )
        assert result == []
        assert lab.distractor == "x-x-x"

    def test_no_duplicates_in_result(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1"
            )
        assert len(result) == len(set(result))


# --- Sentence_Set.do_distractors with locked ---

class TestDoDistractorsLocked:
    def test_locked_label_uses_existing_distractor(self):
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        locked = {1: "dog", 2: "run"}
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict([]), threshold_func, PARAMS, repeats, locked=locked)
        assert "dog" in ss.sentences[0].distractor_sentence
        assert "run" in ss.sentences[0].distractor_sentence

    def test_locked_distractor_added_to_repeats(self):
        ss = make_sentence_set()
        repeats = Repeatcounter(1)
        locked = {1: "dog", 2: "run"}
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict([]), threshold_func, PARAMS, repeats, locked=locked)
        assert "dog" in repeats.banned
        assert "run" in repeats.banned

    def test_locked_distractor_in_local_banned_for_subsequent(self):
        """Locked distractor from label 1 must be in banned when label 2 is processed."""
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        locked = {1: "dog"}
        # Pool only has "dog" — if it's correctly banned, label 2 will fall back to x-x-x
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(["dog"]), threshold_func, PARAMS, repeats, locked=locked)
        # "dog" used for label 1; label 2 can't reuse it, so it should get x-x-x
        words = ss.sentences[0].distractor_sentence.split()
        assert words.count("dog") == 1  # appears only once

    def test_top_n_stores_in_label_options(self):
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        pool = ["dog", "run", "fast", "jump", "red"]
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(pool), threshold_func, PARAMS, repeats, num_options=3)
        for lab in [1, 2]:
            assert lab in ss.label_options
            assert len(ss.label_options[lab]) == 3

    def test_label_options_populated_in_default_mode(self):
        """Even in single-option mode, label_options should be populated."""
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(["dog", "run"]), threshold_func, PARAMS, repeats)
        assert 1 in ss.label_options
        assert 2 in ss.label_options

    def test_default_behavior_produces_non_empty_distractor_sentence(self):
        """With no locked/num_options, distractor_sentence is set as before."""
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(["dog", "run"]), threshold_func, PARAMS, repeats)
        assert ss.sentences[0].distractor_sentence.startswith("x-x-x")
        assert len(ss.sentences[0].distractor_sentence.split()) == 3
