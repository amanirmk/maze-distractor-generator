import sys
import os
import csv
import pytest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_set import Sentence, Sentence_Set
from output import save_longform, save_updated_longform


def make_complete_ss(label_options=None):
    """Build a Sentence_Set with distractors fully assigned (no model needed)."""
    ss = Sentence_Set("1")
    sent = Sentence(["The", "cat", "sat"], [0, 1, 2], "1", "test")
    sent.surprisal = {1: 8.0, 2: 8.0}
    ss.add(sent)
    ss.make_labels()
    sent.distractors = ["x-x-x", "dog", "run"]
    sent.distractor_sentence = "x-x-x dog run"
    ss.label_options = label_options if label_options is not None else {1: ["dog"], 2: ["run"]}
    return ss


def read_longform(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- save_longform ---

class TestSaveLongform:
    def test_skips_first_word_position(self, tmp_path):
        ss = make_complete_ss()
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert all(row["distractor"] != "x-x-x" for row in rows)

    def test_row_count_is_n_minus_one_per_sentence(self, tmp_path):
        ss = make_complete_ss()  # 3-word sentence → 2 non-first positions
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert len(rows) == 2

    def test_prefix_is_words_before_position(self, tmp_path):
        ss = make_complete_ss()
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert rows[0]["prefix"] == "The"
        assert rows[1]["prefix"] == "The cat"

    def test_real_word_matches_sentence(self, tmp_path):
        ss = make_complete_ss()
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert rows[0]["real_word"] == "cat"
        assert rows[1]["real_word"] == "sat"

    def test_distractor_matches_assigned(self, tmp_path):
        ss = make_complete_ss()
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert rows[0]["distractor"] == "dog"
        assert rows[1]["distractor"] == "run"

    def test_options_empty_for_single_option(self, tmp_path):
        ss = make_complete_ss(label_options={1: ["dog"], 2: ["run"]})
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert rows[0]["options"] == ""
        assert rows[1]["options"] == ""

    def test_options_has_alternatives_for_multi(self, tmp_path):
        ss = make_complete_ss(label_options={1: ["dog", "fox", "elk"], 2: ["run", "hop"]})
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        # options for label 1: alternatives beyond first (fox, elk)
        assert "fox" in rows[0]["options"]
        assert "elk" in rows[0]["options"]
        # options for label 2: just "hop"
        assert "hop" in rows[1]["options"]

    def test_rejected_column_empty_on_initial(self, tmp_path):
        ss = make_complete_ss()
        outfile = str(tmp_path / "out.csv")
        save_longform(outfile, {"1": ss})
        rows = read_longform(outfile)
        assert all(row["rejected"] == "" for row in rows)


# --- save_updated_longform ---

ORIGINAL_REVIEW = """\
type,item_num,label,prefix,real_word,distractor,options,rejected
test,1,1,The,cat,dog,,yes
test,1,2,The cat,sat,run,,
"""


class TestSaveUpdatedLongform:
    def test_rejected_row_gets_new_distractor(self, tmp_path):
        original = tmp_path / "review.csv"
        original.write_text(ORIGINAL_REVIEW)
        ss = make_complete_ss(label_options={1: ["fox"], 2: ["run"]})
        # Override distractors to reflect new generation for label 1
        ss.sentences[0].distractors = ["x-x-x", "fox", "run"]
        outfile = str(tmp_path / "updated.csv")
        save_updated_longform(outfile, str(original), {"1": ss})
        rows = read_longform(outfile)
        cat_row = next(r for r in rows if r["real_word"] == "cat")
        assert cat_row["distractor"] == "fox"

    def test_rejected_flag_cleared_after_update(self, tmp_path):
        original = tmp_path / "review.csv"
        original.write_text(ORIGINAL_REVIEW)
        ss = make_complete_ss(label_options={1: ["fox"], 2: ["run"]})
        ss.sentences[0].distractors = ["x-x-x", "fox", "run"]
        outfile = str(tmp_path / "updated.csv")
        save_updated_longform(outfile, str(original), {"1": ss})
        rows = read_longform(outfile)
        cat_row = next(r for r in rows if r["real_word"] == "cat")
        assert cat_row["rejected"] == ""

    def test_non_rejected_row_unchanged(self, tmp_path):
        original = tmp_path / "review.csv"
        original.write_text(ORIGINAL_REVIEW)
        ss = make_complete_ss(label_options={1: ["fox"], 2: ["run"]})
        ss.sentences[0].distractors = ["x-x-x", "fox", "run"]
        outfile = str(tmp_path / "updated.csv")
        save_updated_longform(outfile, str(original), {"1": ss})
        rows = read_longform(outfile)
        sat_row = next(r for r in rows if r["real_word"] == "sat")
        assert sat_row["distractor"] == "run"

    def test_options_column_added_for_multi(self, tmp_path):
        original = tmp_path / "review.csv"
        original.write_text(ORIGINAL_REVIEW)
        ss = make_complete_ss(label_options={1: ["fox", "elk", "owl"], 2: ["run"]})
        ss.sentences[0].distractors = ["x-x-x", "fox", "run"]
        outfile = str(tmp_path / "updated.csv")
        save_updated_longform(outfile, str(original), {"1": ss})
        rows = read_longform(outfile)
        cat_row = next(r for r in rows if r["real_word"] == "cat")
        assert "elk" in cat_row["options"]
        assert "owl" in cat_row["options"]

    def test_all_rows_preserved_in_output(self, tmp_path):
        original = tmp_path / "review.csv"
        original.write_text(ORIGINAL_REVIEW)
        ss = make_complete_ss(label_options={1: ["fox"], 2: ["run"]})
        ss.sentences[0].distractors = ["x-x-x", "fox", "run"]
        outfile = str(tmp_path / "updated.csv")
        save_updated_longform(outfile, str(original), {"1": ss})
        rows = read_longform(outfile)
        assert len(rows) == 2
