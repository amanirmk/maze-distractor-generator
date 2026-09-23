"""Command line for maze-distractors.

maze-distractors items.csv distractors.csv
maze-distractors items.csv items.txt --format ibex --report report.csv
"""

import json
import logging
import math
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from maze_distractors.formats import WRITERS, Format, write_report
from maze_distractors.generation import MISSING, Settings, generate
from maze_distractors.items import read_sentences
from maze_distractors.surprisal import Scorer
from maze_distractors.vocabulary import Vocabulary

DEFAULT_MODEL = "openai-community/gpt2-medium"
_DEFAULTS = Settings()

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def prepare_output_paths(
    items_csv: Path, output: Path, report: Path | None
) -> None:
    """Refuse a path that would overwrite the input or another output,
    and make the directories now rather than after the slow part of the
    run."""
    outputs = [output] if report is None else [output, report]
    for file in outputs:
        if file.resolve() == items_csv.resolve():
            raise ValueError(f"{file} is the input file; it would be lost.")
    if len({file.resolve() for file in outputs}) < len(outputs):
        raise ValueError(f"{report} is both the output and the report.")
    for file in outputs:
        file.parent.mkdir(parents=True, exist_ok=True)


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
            help="Words a distractor may be, one per line. Default: the "
            "curated list for English; otherwise wordfreq's small list, "
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
    settings = Settings(
        min_delta=min_delta,
        min_abs=min_abs,
        max_repeat=max_repeat,
        max_frequency_ratio=max_frequency_ratio,
        break_noun_phrases=break_noun_phrases,
        seed=seed,
    )
    try:
        prepare_output_paths(items_csv, output, report)
        sentences = read_sentences(items_csv)
        vocabulary = Vocabulary.load(language, include, exclude or ())
        scorer = Scorer.from_pretrained(
            model, revision=revision, device=device, bos_token=bos_token
        )
        distracted, chosen = generate(sentences, scorer, vocabulary, settings)
        WRITERS[output_format](output, distracted)
        if report is not None:
            write_report(report, chosen)
    except (ValueError, OSError) as error:
        # A bad path or input, an unknown model or token, or text the
        # tokenizer cannot split into words: the message says which.
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error

    record = {
        "maze_distractors": metadata.version("maze-distractors"),
        # Its word list decides the vocabulary and the order words are tried.
        "wordfreq": metadata.version("wordfreq"),
        "items": str(items_csv),
        "output": str(output),
        "model": model,
        "model_commit": getattr(scorer.model.config, "_commit_hash", None),
        "start_token": scorer.start_token,
        "device": str(scorer.model.device),
        "language": language,
        "include": None if include is None else str(include),
        "exclude": [str(file) for file in exclude or ()],
        "vocabulary_size": len(vocabulary),
        # JSON has no infinity; "inf" is what the option takes.
        "settings": {
            name: "inf" if value == math.inf else value
            for name, value in vars(settings).items()
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
    typer.echo(json.dumps(record, indent=2))


def main() -> None:
    app()
