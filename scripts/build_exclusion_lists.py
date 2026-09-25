"""Rebuild src/maze_distractors/data/first_names.txt and abbreviations.txt,
the words of the curated list left out because readers know them only as
a name ("josh", "tony") or an abbreviation ("pa", "rev"), not as a word.

    uv run python scripts/build_exclusion_lists.py

A word is a candidate if it is a first name in the 1990 US Census lists,
or a USPS state or territory code, or an abbreviated title in the US
Government Publishing Office Style Manual; it is left out by
maze_distractors.capitalization.is_mainly_a_name or
is_mainly_an_abbreviation, from how often SUBTLEX-US has it capitalized
and how often in lower case. The lists are the output of those rules, not
of anyone's judgement. SUBTLEX-US and the Census lists are downloaded
unless --subtlex and --census name copies; neither is redistributed here.

Brysbaert, M., & New, B. (2009). Moving beyond Kucera and Francis.
Behavior Research Methods, 41(4), 977-990.
U.S. Census Bureau (1995). Frequently occurring first names and surnames
from the 1990 census.
U.S. Postal Service. Publication 28, Postal Addressing Standards,
Appendix B.
U.S. Government Publishing Office (2016). Style Manual, sections
9.29-9.32.
"""

import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import Annotated

import typer

from maze_distractors.capitalization import (
    capitalized_shares,
    download_subtlex,
    is_mainly_a_name,
    is_mainly_an_abbreviation,
    lowercase_per_million,
)
from maze_distractors.vocabulary import Vocabulary, packaged_words

DATA = Path(__file__).parents[1] / "src/maze_distractors/data"
CENSUS_URL = "https://www2.census.gov/topics/genealogy/1990surnames/"
CENSUS_SHA256 = {
    "dist.male.first": (
        "0a5078ef6effe3b483d15b0f7f95047662126c9bfb624ecd5e5b978fc0f2470b"
    ),
    "dist.female.first": (
        "bd2f310fc4e5d5e5ea122c9d4342c9821145823118eb20db1647f305ec77b358"
    ),
}
# Publication 28, Appendix B: states, DC, territories, freely associated
# states and the military "states".
USPS_CODES = frozenset((
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc", "as", "gu", "mp", "pr", "vi", "fm", "mh", "pw", "aa",
    "ae", "ap",
))  # fmt: skip
# Style Manual 9.29 (civil titles and the table of military ranks), 9.31
# and 9.32, without their periods.
GPO_TITLES = frozenset((
    "mr", "mrs", "ms", "m", "mm", "messrs", "mlle", "mme", "dr", "hon", "rev", "msgr",
    "rt", "esq", "jr", "sr", "gen", "adm", "ltg", "vadm", "mg", "radm", "rdml", "bg",
    "brig", "col", "capt", "ltc", "cdr", "maj", "lcdr", "cpt", "lt", "ltjg", "ens", "wo",
    "sma", "sgm", "csm", "msg", "sfc", "ssg", "sgt", "cpl", "spc", "pfc", "sn", "sa",
    "cpo", "scpo", "mcpo", "mcpon", "mcpocg", "ccm", "cmsaf", "nco",
))  # fmt: skip


def read_census_names(directory: Path) -> set[str]:
    """Every first name of the Census lists in ``directory``, each file
    checked against the copy the shipped list was built from."""
    names = set()
    for name, sha256 in CENSUS_SHA256.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise SystemExit(f"{name} is not the file the list was built from.")
        names |= {
            line.split()[0].lower()
            for line in data.decode().splitlines()
            if line.strip()
        }
    return names


def download_census(directory: Path) -> Path:
    for name in CENSUS_SHA256:
        with urllib.request.urlopen(CENSUS_URL + name, timeout=60) as response:
            (directory / name).write_bytes(response.read())
    return directory


def main(
    subtlex: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Default: download."),
    ] = None,
    census: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="A directory holding the Census lists. Default: download.",
        ),
    ] = None,
) -> None:
    # The curated list as shipped, not Vocabulary.load(): the lists being
    # rebuilt must not decide which words get judged.
    words = sorted(Vocabulary(packaged_words("curated_word_list.txt")).words)
    with tempfile.TemporaryDirectory() as directory:
        subtlex_file = subtlex or download_subtlex(Path(directory))
        shares = capitalized_shares(subtlex_file)
        lowercase = lowercase_per_million(subtlex_file)
        names = read_census_names(census or download_census(Path(directory)))
    lists = {
        "first_names.txt": [
            w
            for w in words
            if w in names and is_mainly_a_name(shares.get(w), lowercase.get(w))
        ],
        "abbreviations.txt": [
            w
            for w in words
            if w in USPS_CODES | GPO_TITLES
            and is_mainly_an_abbreviation(shares.get(w), lowercase.get(w))
        ],
    }
    for file, listed in lists.items():
        (DATA / file).write_text("\n".join(listed) + "\n", encoding="utf-8")
        typer.echo(f"{len(listed)} words written to {DATA / file}")


if __name__ == "__main__":
    typer.run(main)
