"""Write sentences with their distractors, and the per-position report."""

import csv
import json
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

from maze_distractors.generation import ChosenDistractor, SentenceDistractors


def write_csv(file: Path, distracted: Sequence[SentenceDistractors]) -> None:
    """One row per sentence: what was read, and its distractors."""
    with file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["type", "item_num", "sentence", "distractors", "labels"]
        )
        for d in distracted:
            writer.writerow(
                [
                    d.sentence.tag,
                    d.sentence.item,
                    " ".join(d.sentence.words),
                    " ".join(d.distractors),
                    " ".join(d.sentence.labels),
                ]
            )


def write_ibex(file: Path, distracted: Sequence[SentenceDistractors]) -> None:
    """One line per sentence, to paste into the ``items`` array of an Ibex
    Maze experiment, in the layout A-Maze writes."""
    with file.open("w", encoding="utf-8") as f:
        for d in distracted:
            item = d.sentence.item.replace("\\", "\\\\").replace("'", "\\'")
            f.write(
                f"[[{json.dumps(d.sentence.tag)}, '{item}'], \"Maze\", "
                f"{{s:{json.dumps(' '.join(d.sentence.words))}, "
                f"a:{json.dumps(' '.join(d.distractors))}}}], \n"
            )


def write_jspsych(
    file: Path, distracted: Sequence[SentenceDistractors]
) -> None:
    """A JavaScript module exporting ``stimuli``, for jsPsych."""
    stimuli = [
        {
            "item_type": d.sentence.tag,
            "id": d.sentence.item,
            "sentence": " ".join(d.sentence.words),
            "distractor": " ".join(d.distractors),
            "labels": " ".join(d.sentence.labels),
        }
        for d in distracted
    ]
    file.write_text(
        f"export const stimuli = {json.dumps(stimuli, indent=2)};\n",
        encoding="utf-8",
    )


class Format(StrEnum):
    CSV = "csv"
    IBEX = "ibex"
    JSPSYCH = "jspsych"


WRITERS: dict[Format, Callable[[Path, Sequence[SentenceDistractors]], None]] = {
    Format.CSV: write_csv,
    Format.IBEX: write_ibex,
    Format.JSPSYCH: write_jspsych,
}


def write_report(file: Path, chosen: Sequence[ChosenDistractor]) -> None:
    """One row per target word: the surprisal threshold it set, the
    distractor's surprisal there, and whether it met it."""
    with file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "type",
                "item_num",
                "label",
                "position",
                "word",
                "distractor",
                "threshold",
                "surprisal",
                "met",
            ]
        )
        for p in chosen:
            writer.writerow(
                [
                    p.sentence.tag,
                    p.sentence.item,
                    p.sentence.labels[p.index],
                    p.index,
                    p.sentence.words[p.index],
                    p.distractor,
                    f"{p.threshold:.3f}",
                    f"{p.surprisal:.3f}",
                    p.met,
                ]
            )
