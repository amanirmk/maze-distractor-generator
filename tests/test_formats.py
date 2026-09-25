import csv
import json
import math
import re

from maze_distractors.formats import (
    write_csv,
    write_ibex,
    write_jspsych,
    write_report,
)
from maze_distractors.generation import (
    MISSING,
    PLACEHOLDER,
    ChosenDistractor,
    SentenceDistractors,
)
from maze_distractors.items import Sentence

SENTENCE = Sentence(
    "sub_rel", "3", ("The", "cat", 'said "hi".'), ("pre", "noun", "verb")
)
DISTRACTED = [SentenceDistractors(SENTENCE, (PLACEHOLDER, "rug", 'mild "os".'))]


def test_csv_is_the_input_plus_distractors(tmp_path):
    file = tmp_path / "out.csv"
    write_csv(file, DISTRACTED)
    with file.open(newline="") as f:
        (row,) = csv.DictReader(f)
    assert row == {
        "type": "sub_rel",
        "item_num": "3",
        "sentence": 'The cat said "hi".',
        "distractors": 'x-x-x rug mild "os".',
        "labels": "pre noun verb",
    }


def test_ibex_lines_have_a_mazes_layout_with_quotes_escaped(tmp_path):
    file = tmp_path / "items.txt"
    write_ibex(file, DISTRACTED)
    assert file.read_text() == (
        '[["sub_rel", \'3\'], "Maze", {s:"The cat said \\"hi\\".", '
        'a:"x-x-x rug mild \\"os\\"."}], \n'
    )
    # What a reader of A-Maze's own files looks for.
    assert re.search(r"\[\[\"(.+)\", '(.+)'\],", file.read_text()).groups() == (
        "sub_rel",
        "3",
    )


def test_jspsych_is_a_module_exporting_stimuli(tmp_path):
    file = tmp_path / "stimuli.js"
    write_jspsych(file, DISTRACTED)
    text = file.read_text()
    assert text.startswith("export const stimuli = ")
    assert text.endswith(";\n")
    (stimulus,) = json.loads(text.removeprefix("export const stimuli = ")[:-2])
    assert stimulus == {
        "item_type": "sub_rel",
        "id": "3",
        "sentence": 'The cat said "hi".',
        "distractor": 'x-x-x rug mild "os".',
        "labels": "pre noun verb",
    }


def test_the_report_has_a_row_per_position(tmp_path):
    file = tmp_path / "report.csv"
    positions = [
        ChosenDistractor(SENTENCE, 1, "rug", threshold=25.0, surprisal=27.1234),
        ChosenDistractor(
            SENTENCE, 2, 'mild "os".', threshold=31.5, surprisal=30.0
        ),
    ]
    write_report(file, positions)
    with file.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == {
        "type": "sub_rel",
        "item_num": "3",
        "label": "noun",
        "position": "1",
        "word": "cat",
        "distractor": "rug",
        "threshold": "25.000000",
        "surprisal": "27.123400",
        "met": "True",
    }
    assert rows[1]["met"] == "False"


def test_a_position_without_a_distractor_is_reported_as_such(tmp_path):
    file = tmp_path / "report.csv"
    missing = ChosenDistractor(
        SENTENCE, 1, MISSING, threshold=25.0, surprisal=math.nan
    )
    write_report(file, [missing])
    with file.open(newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["distractor"] == MISSING
    assert row["surprisal"] == "nan"
    assert row["met"] == "False"


def test_csv_for_an_amaze_input_is_amazes_layout(tmp_path):
    file = tmp_path / "out.csv"
    write_csv(file, DISTRACTED, amaze=True)
    (row,) = file.read_text().splitlines()
    assert row.startswith("sub_rel;3;")
    assert row.count(";") == 4
    assert "type" not in row
