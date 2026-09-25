"""Sentences to find distractors for. Those of one item share distractors
wherever they share a label."""

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from maze_distractors.punctuation import strip_punctuation

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("type", "item_num", "sentence")
LABEL_COLUMNS = ("labels", "label")


@dataclass(frozen=True)
class InputLayout:
    """How a file is laid out: with a header line naming the columns
    (comma- or semicolon-separated), or as A-Maze's own input -- the same
    columns by position, semicolon-separated, with no header -- so that
    A-Maze's files work as they are."""

    headed: bool
    delimiter: str


def _column_name(cell: str) -> str:
    return cell.strip().lower()


def _first_line_cells(first: str, delimiter: str) -> set[str]:
    # Parsed as CSV, not split: R's write.csv quotes every column name.
    return {
        _column_name(cell)
        for cell in next(csv.reader([first], delimiter=delimiter), [])
    }


def detect_layout(file: Path) -> InputLayout:
    with file.open(encoding="utf-8-sig", newline="") as f:
        first = f.readline()
    for delimiter in (",", ";"):
        if set(REQUIRED_COLUMNS) <= _first_line_cells(first, delimiter):
            return InputLayout(headed=True, delimiter=delimiter)
    # A first line that names any column is a header, even with the other
    # names spelled differently: read as A-Maze's rows, it would be a
    # sentence of its own, shown to participants.
    for delimiter in (",", ";"):
        named = _first_line_cells(first, delimiter)
        known = named & {*REQUIRED_COLUMNS, *LABEL_COLUMNS}
        if known:
            missing = [c for c in REQUIRED_COLUMNS if c not in named]
            raise ValueError(
                f"The first line of {file} looks like a header (it names "
                f"{', '.join(sorted(known))}) but has no column "
                f"{', '.join(missing)}. The columns are "
                f"{', '.join(REQUIRED_COLUMNS)} and optionally labels."
            )
    if ";" in first:
        return InputLayout(headed=False, delimiter=";")
    raise ValueError(
        f"{file} has neither a header line naming {', '.join(REQUIRED_COLUMNS)} "
        "nor A-Maze's semicolon-separated rows."
    )


@dataclass(frozen=True)
class Sentence:
    tag: str  # passed through to the output untouched
    item: str
    words: tuple[str, ...]
    labels: tuple[str, ...]


def read_sentences(file: Path) -> list[Sentence]:
    """Read the sentences of ``file``: the columns ``type``, ``item_num``,
    ``sentence`` and optionally ``labels`` (one label per word,
    space-separated), one sentence per row, as a CSV with a header or as
    A-Maze's semicolon-separated file without one (``InputLayout``).
    Column names are matched ignoring case and spaces, and ``label`` is
    taken for ``labels``. Without labels a word is labelled by its
    position, so sentences of one item share distractors position by
    position."""
    layout = detect_layout(file)
    # (tag, item_num, sentence, labels or None when there is no labels
    # cell, line number)
    rows: list[tuple[str, str, str, str | None, int]] = []
    # utf-8-sig: spreadsheets put a byte-order mark before the first column
    # name.
    with file.open(encoding="utf-8-sig", newline="") as f:
        if layout.headed:
            reader = csv.DictReader(f, delimiter=layout.delimiter)
            reader.fieldnames = [
                _column_name(name) for name in reader.fieldnames or ()
            ]
            label_column = next(
                (c for c in LABEL_COLUMNS if c in reader.fieldnames), None
            )
            for row in reader:
                if None in row or None in row.values():
                    raise ValueError(
                        f"Line {reader.line_num} of {file} has more or fewer "
                        "cells than the header."
                    )
                rows.append(
                    (
                        row["type"],
                        row["item_num"],
                        row["sentence"],
                        row[label_column] if label_column else None,
                        reader.line_num,
                    )
                )
        else:
            amaze = csv.reader(f, delimiter=";", quotechar='"')
            for row in amaze:
                # A spreadsheet export can end every row in empty cells.
                while len(row) > 4 and not row[-1].strip():
                    row.pop()
                if len(row) not in (3, 4):
                    raise ValueError(
                        f"Line {amaze.line_num} of {file} has {len(row)} "
                        "cells; A-Maze's rows are type;item_num;sentence, "
                        "with labels as an optional fourth."
                    )
                rows.append(
                    (
                        row[0],
                        row[1],
                        row[2],
                        row[3] if len(row) == 4 else None,
                        amaze.line_num,
                    )
                )
    _warn_of_dropped_quotes(file, layout.delimiter)
    unlabelled = sum(1 for *_, labels, _ in rows if not (labels or "").strip())
    if unlabelled:
        logger.warning(
            "%s: %d of %d sentence(s) have no labels, so their words are "
            "labelled by position and sentences of an item share "
            "distractors position by position.",
            file,
            unlabelled,
            len(rows),
        )
    sentences = [
        _sentence(tag, item, text, labels or "", f"Line {line} of {file}")
        for tag, item, text, labels, line in rows
    ]
    by_item: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        by_item.setdefault(sentence.item, []).append(sentence)
    for of_item in by_item.values():
        validate_item(of_item)
    return sentences


def _warn_of_dropped_quotes(file: Path, delimiter: str) -> None:
    """Name the lines where a cell opens with a quote mark that closes
    before the cell ends, as in ``"Stop," he said.``: read as CSV quoting,
    as A-Maze reads it too, so those quote marks are not shown. A cell a
    spreadsheet quoted is well-formed CSV and is not named."""
    malformed = []
    with file.open(encoding="utf-8-sig", newline="") as f:
        for number, line in enumerate(f, start=1):
            try:
                next(csv.reader([line], delimiter=delimiter, strict=True), None)
            except csv.Error:
                malformed.append(number)
    if malformed:
        logger.warning(
            "%s, line(s) %s: a cell opens with a quote mark that closes "
            "before the cell ends. It is read as CSV quoting, as in A-Maze, "
            "so those quote marks are not shown. To show them, put the "
            'whole cell in quotes and double the quote marks inside it ("").',
            file,
            ", ".join(map(str, malformed)),
        )


def _sentence(
    tag: str, item: str, text: str, labels_text: str, where: str
) -> Sentence:
    if "\n" in text or "\r" in text:
        # A cell that opens with a quote mark and never closes it runs on
        # through the rows after it, which would vanish into this one.
        raise ValueError(
            f"{where}: the sentence runs over more than one line, usually "
            "an opening quote mark with no closing one. CSV quoting applies: "
            'to show a quote mark in a sentence, double it ("").'
        )
    words = tuple(text.split())
    labels = tuple(labels_text.split()) or tuple(
        str(i) for i in range(len(words))
    )
    sentence = Sentence(tag.strip(), item.strip(), words, labels)
    if not sentence.item:
        raise ValueError(f"{where} has no item_num.")
    validate_sentence(sentence)
    return sentence


def validate_sentence(sentence: Sentence) -> None:
    text = " ".join(sentence.words)
    if not sentence.words:
        raise ValueError(f"Item {sentence.item!r} has an empty sentence.")
    wordless = [w for w in sentence.words if not strip_punctuation(w)]
    if wordless:
        raise ValueError(
            f"{wordless} in {text!r} would be shown alone, without a letter "
            "or digit; join it to the word beside it."
        )
    if len(sentence.labels) != len(sentence.words):
        raise ValueError(
            f"{len(sentence.labels)} labels for {len(sentence.words)} words "
            f"in {text!r}."
        )
    counts = Counter(sentence.labels)
    repeated = [label for label, n in counts.items() if n > 1]
    if repeated:
        raise ValueError(f"Label(s) {repeated} repeat within {text!r}.")


def validate_item(sentences: list[Sentence]) -> None:
    """A sentence's first word never gets a distractor, so a label cannot
    be first in one sentence of an item and later in another."""
    first = {sentence.labels[0] for sentence in sentences}
    later = {label for s in sentences for label in s.labels[1:]}
    if first & later:
        raise ValueError(
            f"Item {sentences[0].item!r} uses the label(s) "
            f"{sorted(first & later)} both for a first word and a later one."
        )
    # Sentences are put in one item to share distractors; one that shares
    # no label with the rest is usually a labels cell left blank.
    for sentence in sentences:
        others = {
            label for s in sentences if s is not sentence for label in s.labels
        }
        if others and not others & set(sentence.labels):
            raise ValueError(
                f"{' '.join(sentence.words)!r} shares no label with the other "
                f"sentence(s) of item {sentence.item!r}."
            )
