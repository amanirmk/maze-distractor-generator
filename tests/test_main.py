import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import read_rejection_file
from limit_repeats import Repeatcounter

REVIEW_CSV = """\
type,item_num,label,prefix,real_word,distractor,options,rejected
passage,1,1,Tyrannosaurus,rex,whoa,,
passage,1,2,Tyrannosaurus rex,is,sale,,yes
passage,2,1,For,such,shop,,
passage,2,2,For such,a,knew,,yes
"""

REVIEW_CSV_NONEMPTY_REJECTED = """\
type,item_num,label,prefix,real_word,distractor,options,rejected
passage,1,1,The,cat,dog,,x
passage,1,2,The cat,sat,run,,maybe
"""


class TestReadRejectionFile:
    def test_rejected_items_identified(self, tmp_path):
        f = tmp_path / "review.csv"
        f.write_text(REVIEW_CSV)
        locked, rejected = read_rejection_file(str(f))
        assert "1" in rejected
        assert "2" in rejected

    def test_locked_excludes_rejected_positions(self, tmp_path):
        f = tmp_path / "review.csv"
        f.write_text(REVIEW_CSV)
        locked, rejected = read_rejection_file(str(f))
        # item 1: label 1 is accepted (locked), label 2 is rejected
        assert 1 in locked.get("1", {})
        assert 2 not in locked.get("1", {})

    def test_locked_distractor_value_preserved(self, tmp_path):
        f = tmp_path / "review.csv"
        f.write_text(REVIEW_CSV)
        locked, rejected = read_rejection_file(str(f))
        assert locked["1"][1] == "whoa"

    def test_label_type_is_int(self, tmp_path):
        f = tmp_path / "review.csv"
        f.write_text(REVIEW_CSV)
        locked, _ = read_rejection_file(str(f))
        for item_locked in locked.values():
            for label in item_locked:
                assert isinstance(label, int)

    def test_any_nonempty_rejected_value_counts(self, tmp_path):
        f = tmp_path / "review.csv"
        f.write_text(REVIEW_CSV_NONEMPTY_REJECTED)
        locked, rejected = read_rejection_file(str(f))
        assert "1" in rejected
        assert 1 not in locked.get("1", {})  # label 1 rejected ("x")
        assert 2 not in locked.get("1", {})  # label 2 rejected ("maybe")

    def test_item_with_all_accepted_not_in_rejected(self, tmp_path):
        csv_content = """\
type,item_num,label,prefix,real_word,distractor,options,rejected
passage,3,1,Word,two,dog,,
passage,3,2,Word two,three,run,,
"""
        f = tmp_path / "review.csv"
        f.write_text(csv_content)
        locked, rejected = read_rejection_file(str(f))
        assert "3" not in rejected
        assert "3" in locked
        assert locked["3"][1] == "dog"
        assert locked["3"][2] == "run"
