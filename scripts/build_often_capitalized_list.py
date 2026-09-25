"""Rebuild src/maze_distractors/data/often_capitalized.txt, the words held
to their surprisal threshold capitalized as well as shown ("josh",
"oxford", "academy"): shown in lower case, a reader may still take one for
the name, which the model, scoring the lower case, does not.

    uv run python scripts/build_often_capitalized_list.py
    uv run python scripts/build_often_capitalized_list.py --scores scores.csv
    uv run python scripts/build_often_capitalized_list.py --words w.txt ...

Every word of the curated list (or, with --words, of your own list, whose
result goes to --out) is judged by
maze_distractors.capitalization.is_often_capitalized, from how often
SUBTLEX-US has it capitalized and how much gpt2-medium prefers it so. The
list is the output of that rule, not of anyone's judgement. SUBTLEX-US is
downloaded unless --subtlex names a copy of its text version; it is not
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

from maze_distractors.capitalization import (
    capital_preferences,
    capitalized_shares,
    is_often_capitalized,
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
    Path(__file__).parents[1]
    / "src/maze_distractors/data/often_capitalized.txt"
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


def _output_for(words_file: Path | None, out: Path | None) -> Path:
    """Where the list goes: the shipped file for the curated list. For a
    user's own, a file of theirs: not the shipped list, whose names it
    would replace, and not the word list it is built from."""
    if words_file is None:
        return out or LIST_FILE
    if out is None or out.resolve() == LIST_FILE.resolve():
        raise typer.BadParameter(
            "with --words, give --out a file of your own: the shipped "
            "often-capitalized list is built from the curated words only",
            param_hint="--out",
        )
    if out.resolve() == words_file.resolve():
        raise typer.BadParameter(
            "--out would replace the --words list with part of itself",
            param_hint="--out",
        )
    return out


def main(
    words_file: Annotated[
        Path | None,
        typer.Option(
            "--words",
            exists=True,
            dir_okay=False,
            help="Words to judge, one per line. Default: the curated list.",
        ),
    ] = None,
    subtlex: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Default: download."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help="Default: the shipped list, which --words may not replace.",
        ),
    ] = None,
    scores: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Also write every word's numbers."),
    ] = None,
) -> None:
    out = _output_for(words_file, out)
    # The curated list as shipped, not Vocabulary.load(): the list being
    # rebuilt must not decide which words get judged. Or a user's own
    # list, for --include, whose result they pass with --exclude.
    source = (
        set(words_file.read_text(encoding="utf-8-sig").split())
        if words_file
        else packaged_words("curated_word_list.txt")
    )
    words = sorted(Vocabulary(source).words)
    with tempfile.TemporaryDirectory() as directory:
        shares = capitalized_shares(
            subtlex or download_subtlex(Path(directory))
        )
    # On the CPU wherever it runs: another device's arithmetic can move a
    # word that sits near a threshold.
    scorer = Scorer.from_pretrained(
        MODEL, revision=MODEL_REVISION, device="cpu"
    )
    scored = capital_preferences(scorer, words)
    listed = [
        w for w in words if is_often_capitalized(shares.get(w), scored[w])
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
