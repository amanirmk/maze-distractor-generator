"""Rebuild src/maze_distractors/data/proper_nouns.txt, the words left out
of the vocabulary because they are mainly proper nouns ("josh", "oxford",
"jack"): in lower case they are not the word a reader knows.

    uv run python scripts/build_proper_noun_list.py
    uv run python scripts/build_proper_noun_list.py --scores scores.csv

Every word of the curated list is judged by
maze_distractors.proper_nouns.is_likely_proper_noun, from how often SUBTLEX-US
has it capitalized and how much gpt2-medium prefers it so. The list is the
output of that rule, not of anyone's judgement. SUBTLEX-US is downloaded
unless --subtlex names a copy of its text version; it is not
redistributed here.

Brysbaert, M., & New, B. (2009). Moving beyond Kucera and Francis.
Behavior Research Methods, 41(4), 977-990.
"""

import csv
import hashlib
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Annotated

import typer

from maze_distractors.proper_nouns import (
    capital_preferences,
    capitalized_shares,
    is_likely_proper_noun,
)
from maze_distractors.surprisal import Scorer
from maze_distractors.vocabulary import Vocabulary, packaged_words

# Pinned, so that a rebuild differs from the shipped list only through a
# change to the rule.
MODEL = "openai-community/gpt2-medium"
MODEL_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
SUBTLEX_URL = (
    "https://www.ugent.be/pp/experimentele-psychologie/en/research/"
    "documents/subtlexus/subtlexus2.zip"
)
SUBTLEX_SHA256 = (
    "67e595da1b399d2a21e25a1466a8d9f242f21a219c6bbbd3e089c4779ad83856"
)
SUBTLEX_MEMBER = "SUBTLEXus74286wordstextversion.txt"
LIST_FILE = (
    Path(__file__).parents[1] / "src/maze_distractors/data/proper_nouns.txt"
)


def download_subtlex(directory: Path) -> Path:
    with urllib.request.urlopen(SUBTLEX_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SUBTLEX_SHA256:
        raise SystemExit(
            f"{SUBTLEX_URL} is not the file the list was built from."
        )
    archive = zipfile.ZipFile(io.BytesIO(data))
    return Path(archive.extract(SUBTLEX_MEMBER, directory))


def main(
    subtlex: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Default: download."),
    ] = None,
    out: Annotated[Path, typer.Option(dir_okay=False)] = LIST_FILE,
    scores: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Also write every word's numbers."),
    ] = None,
) -> None:
    # The curated list as shipped, not Vocabulary.load(): the list being
    # rebuilt must not decide which words get judged.
    words = sorted(Vocabulary(packaged_words("curated_word_list.txt")).words)
    with tempfile.TemporaryDirectory() as directory:
        shares = capitalized_shares(
            subtlex or download_subtlex(Path(directory))
        )
    scorer = Scorer.from_pretrained(MODEL, revision=MODEL_REVISION)
    scored = capital_preferences(scorer, words)
    listed = [
        w for w in words if is_likely_proper_noun(shares.get(w), scored[w])
    ]
    out.write_text("\n".join(listed) + "\n", encoding="utf-8")
    typer.echo(f"{len(listed)} of {len(words)} words written to {out}")
    if scores is not None:
        with scores.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["word", "capitalized_share", "model_score", "listed"]
            )
            for w in words:
                share = "" if w not in shares else f"{shares[w]:.3f}"
                writer.writerow(
                    [w, share, f"{scored[w]:.2f}", w in set(listed)]
                )


if __name__ == "__main__":
    typer.run(main)
