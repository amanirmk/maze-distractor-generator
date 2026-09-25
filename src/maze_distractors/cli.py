"""Command line for maze-distractors.

maze-distractors items.csv distractors.csv
maze-distractors items.csv items.txt --format ibex --report report.csv
"""

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from importlib import metadata, resources
from pathlib import Path
from typing import Annotated

import typer

from maze_distractors.formats import WRITERS, Format, write_csv, write_report
from maze_distractors.generation import MISSING, Settings, generate
from maze_distractors.items import detect_layout, read_sentences
from maze_distractors.surprisal import Scorer
from maze_distractors.vocabulary import Vocabulary

DEFAULT_MODEL = "openai-community/gpt2-medium"
_DEFAULTS = Settings()

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def prepare_output_paths(
    inputs: Sequence[Path], output: Path, report: Path | None
) -> None:
    """Refuse a path that would overwrite an input (the items or a word
    list) or another output, and make the directories now rather than
    after the slow part of the run."""
    outputs = [output] if report is None else [output, report]
    for file in outputs:
        if any(file.resolve() == given.resolve() for given in inputs):
            raise ValueError(f"{file} is an input file; it would be lost.")
    if len({file.resolve() for file in outputs}) < len(outputs):
        raise ValueError(f"{report} is both the output and the report.")
    for file in outputs:
        file.parent.mkdir(parents=True, exist_ok=True)


def _sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


PACKAGED_LISTS = (
    "curated_word_list.txt",
    "exclude.txt",
    "often_capitalized.txt",
    "first_names.txt",
    "abbreviations.txt",
    "noun_phrase_breakers.txt",
)


def _json_number(value: object) -> object:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _code_commit() -> str | None:
    """The commit the running code came from, as the installer recorded it
    for a git install (``uvx --from git+...``, a pinned tag). None for any
    other install (an editable clone, a local path): none is on record."""
    try:
        direct = json.loads(
            metadata.distribution("maze-distractors").read_text(
                "direct_url.json"
            )
            or "{}"
        )
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return None
    return direct.get("vcs_info", {}).get("commit_id")


@app.command()
def generate_distractors(
    items_csv: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Columns type, item_num, sentence and optionally labels.",
        ),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    output_format: Annotated[Format, typer.Option("--format")] = Format.CSV,
    report: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help="Also write each position's threshold and surprisal here.",
        ),
    ] = None,
    model: Annotated[
        str, typer.Option(help="A HuggingFace causal language model.")
    ] = DEFAULT_MODEL,
    revision: Annotated[
        str | None, typer.Option(help="Model commit, tag or branch.")
    ] = None,
    bos_token: Annotated[
        str | None,
        typer.Option(
            help="Token that starts every sequence. Default: the tokenizer's."
        ),
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="Default: cuda, mps or cpu.")
    ] = None,
    language: Annotated[
        str, typer.Option(help="wordfreq language of the word frequencies.")
    ] = "en",
    include: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Words a distractor may be, one per line, replacing the "
            "built-in list. For English the built-in exclusions are still "
            "taken out, with a warning naming them. Default: "
            "the curated list for English; otherwise wordfreq's small list, "
            "unscreened, with a warning.",
        ),
    ] = None,
    exclude: Annotated[
        list[Path] | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="More words to exclude, one per line. Repeatable.",
        ),
    ] = None,
    often_capitalized: Annotated[
        list[Path] | None,
        typer.Option(
            "--often-capitalized",
            exists=True,
            dir_okay=False,
            help="More words to score capitalized too, as the built-in "
            "list's are, one per line (scripts/build_often_capitalized_"
            "list.py --words makes one for an --include list). Repeatable.",
        ),
    ] = None,
    min_delta: Annotated[
        float, typer.Option(help="Bits above the target word's surprisal.")
    ] = _DEFAULTS.min_delta,
    min_abs: Annotated[
        float, typer.Option(help="Lowest surprisal threshold, in bits.")
    ] = _DEFAULTS.min_abs,
    max_repeat: Annotated[
        int,
        typer.Option(min=0, help="Uses of one distractor overall; 0 = any."),
    ] = _DEFAULTS.max_repeat,
    max_frequency_ratio: Annotated[
        float,
        typer.Option(
            min=1,
            help="Times rarer or more common than the target word a "
            "distractor may be; inf for any.",
        ),
    ] = _DEFAULTS.max_frequency_ratio,
    min_candidates: Annotated[
        int,
        typer.Option(
            min=1,
            help="Words to try at least, widening the length match if "
            "needed; closer lengths are tried first.",
        ),
    ] = _DEFAULTS.min_candidates,
    break_noun_phrases: Annotated[
        bool,
        typer.Option(
            help="After a determiner, try only words that cannot continue "
            "the noun phrase (English only).",
        ),
    ] = _DEFAULTS.break_noun_phrases,
    seed: Annotated[
        int, typer.Option(help="Fixes the order words are tried in.")
    ] = _DEFAULTS.seed,
) -> None:
    """Choose Maze distractors for the sentences of ITEMS_CSV and write them
    to OUTPUT. A JSON record of the run goes to standard output."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = Settings(
            min_delta=min_delta,
            min_abs=min_abs,
            max_repeat=max_repeat,
            max_frequency_ratio=max_frequency_ratio,
            min_candidates=min_candidates,
            break_noun_phrases=break_noun_phrases,
            seed=seed,
        )
        word_lists = [
            *([include] if include else []),
            *(exclude or ()),
            *(often_capitalized or ()),
        ]
        prepare_output_paths([items_csv, *word_lists], output, report)
        sentences = read_sentences(items_csv)
        vocabulary = Vocabulary.load(
            language, include, exclude or (), often_capitalized or ()
        )
        scorer = Scorer.from_pretrained(
            model, revision=revision, device=device, bos_token=bos_token
        )
        distracted, chosen = generate(sentences, scorer, vocabulary, settings)
        # Both files are written beside their final names and moved into
        # place together, so a failure never leaves new distractors next to
        # an earlier run's report.
        # Beside the file itself, not the name given: a symlink (into an
        # Ibex project, say) is then written through rather than replaced.
        finals = {path: path.resolve() for path in (output, report) if path}
        parts = {
            path: final.with_name(f".{final.name}.part")
            for path, final in finals.items()
        }
        try:
            if output_format is Format.CSV:
                # An A-Maze input gets A-Maze's output layout back.
                amaze = not detect_layout(items_csv).headed
                write_csv(parts[output], distracted, amaze=amaze)
            else:
                WRITERS[output_format](parts[output], distracted)
            if report is not None:
                write_report(parts[report], chosen)
            for path, part in parts.items():
                part.replace(finals[path])
        finally:
            for part in parts.values():
                part.unlink(missing_ok=True)
    except (ValueError, OSError) as error:
        # A bad path or input, an unknown model or token, or text the
        # tokenizer cannot split into words: the message says which.
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error

    record = {
        "maze_distractors": metadata.version("maze-distractors"),
        # The version stays put between releases; the commit does not.
        "maze_distractors_commit": _code_commit(),
        # The built-in lists by content, since a clone can edit them.
        "word_lists_sha256": {
            name: hashlib.sha256(
                (
                    resources.files("maze_distractors") / "data" / name
                ).read_bytes()
            ).hexdigest()
            for name in PACKAGED_LISTS
        },
        # Its word list decides the vocabulary and the order words are tried.
        "wordfreq": metadata.version("wordfreq"),
        # The scoring stack: a change in any of it can move a borderline
        # choice, and `uvx` resolves the newest rather than the lockfile.
        "transformers": metadata.version("transformers"),
        "tokenizers": metadata.version("tokenizers"),
        "torch": metadata.version("torch"),
        "items": str(items_csv),
        "items_sha256": _sha256(items_csv),
        "output": str(output),
        "model": model,
        "model_commit": getattr(scorer.model.config, "_commit_hash", None),
        "start_token": scorer.start_token,
        "device": str(scorer.model.device),
        "language": language,
        "include": None if include is None else str(include),
        "exclude": [str(file) for file in exclude or ()],
        "often_capitalized": [str(file) for file in often_capitalized or ()],
        # The word lists by content, since a path can be edited in place.
        "include_sha256": None if include is None else _sha256(include),
        "exclude_sha256": [_sha256(file) for file in exclude or ()],
        "often_capitalized_sha256": [
            _sha256(file) for file in often_capitalized or ()
        ],
        "vocabulary_size": len(vocabulary),
        # JSON has no infinity; "inf" and "-inf" are what the options take.
        "settings": {
            name: _json_number(value) for name, value in vars(settings).items()
        },
        "positions": len(chosen),
        "positions_short_of_threshold": sum(
            d.distractor != MISSING and not d.met for d in chosen
        ),
        "positions_without_distractor": sum(
            d.distractor == MISSING for d in chosen
        ),
    }
    if record["positions_without_distractor"]:
        logger.warning(
            "%d position(s) have no distractor and show %s: no word of the "
            "vocabulary could be tried there (see the warnings above).",
            record["positions_without_distractor"],
            MISSING,
        )
    typer.echo(json.dumps(record, indent=2, allow_nan=False))


def main() -> None:
    app()
