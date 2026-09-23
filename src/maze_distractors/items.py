"""Sentences to find distractors for. Those of one item share distractors
wherever they share a label."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from maze_distractors.punctuation import strip_punctuation

REQUIRED_COLUMNS = ("type", "item_num", "sentence")


@dataclass(frozen=True)
class Sentence:
    tag: str  # passed through to the output untouched
    item: str
    words: tuple[str, ...]
    labels: tuple[str, ...]


def read_sentences(file: Path) -> list[Sentence]:
    """Read a CSV with the columns ``type``, ``item_num``, ``sentence`` and
    optionally ``labels`` (one label per word, space-separated), one
    sentence per row. Without labels a word is labelled by its position, so
    sentences of one item share distractors position by position."""
    sentences: list[Sentence] = []
    # utf-8-sig: spreadsheets put a byte-order mark before the first column
    # name.
    with file.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{file} lacks the column(s) {sorted(missing)}.")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(
                    f"Line {reader.line_num} of {file} has more or fewer "
                    "cells than the header."
                )
            words = tuple(row["sentence"].split())
            labels = tuple((row.get("labels") or "").split()) or tuple(
                str(i) for i in range(len(words))
            )
            sentence = Sentence(
                row["type"].strip(), row["item_num"].strip(), words, labels
            )
            if not sentence.item:
                raise ValueError(
                    f"Line {reader.line_num} of {file} has no item_num."
                )
            validate_sentence(sentence)
            sentences.append(sentence)
    by_item: dict[str, list[Sentence]] = {}
    for sentence in sentences:
        by_item.setdefault(sentence.item, []).append(sentence)
    for of_item in by_item.values():
        validate_item(of_item)
    return sentences


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
