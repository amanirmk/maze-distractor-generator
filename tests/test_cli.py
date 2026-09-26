import csv
import hashlib
import json
from importlib import metadata
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import build_model
from maze_distractors import cli
from maze_distractors.cli import app
from maze_distractors.generation import Settings
from maze_distractors.surprisal import Scorer

ITEMS = (
    "type,item_num,sentence,labels\n"
    "sub_rel,3,The cat who the dog scared hid.,a b who art noun verb main\n"
    "obj_rel,3,The dog who scared the cat sniffed.,a b who verb art noun main\n"
)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, tokenizer):
    directory = tmp_path_factory.mktemp("model")
    build_model(len(tokenizer)).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def run(*args, exit_code=0):
    result = CliRunner().invoke(app, [str(arg) for arg in args])
    assert result.exit_code == exit_code, result.output
    return result


def test_a_run_writes_the_output_the_report_and_a_record(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("garden\n")
    output = tmp_path / "nested" / "out.csv"
    report = tmp_path / "report.csv"
    result = run(
        items, output, "--report", report, "--model", model_dir,
        "--device", "cpu", "--min-abs", 20, "--seed", 7, "--exclude", exclude,
    )  # fmt: skip

    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["type"] for row in rows] == ["sub_rel", "obj_rel"]
    assert all(row["distractors"].startswith("x-x-x ") for row in rows)
    with report.open(newline="") as f:
        reported = list(csv.DictReader(f))
    assert len(reported) == 12

    record = json.loads(result.stdout)
    assert record["wordfreq"] == metadata.version("wordfreq")
    assert record["model"] == str(model_dir)
    assert record["start_token"] == "<|endoftext|>"
    assert record["device"] == "cpu"
    assert record["settings"] == {
        "min_delta": 10.0,
        "min_abs": 20.0,
        "max_repeat": 1,
        "max_frequency_ratio": 10.0,
        "min_candidates": 10,
        "break_noun_phrases": True,
        "seed": 7,
    }
    assert record["items"] == str(items)
    assert record["include"] is None
    assert record["exclude"] == [str(exclude)]
    assert record["include_sha256"] is None
    assert record["often_capitalized"] == []
    assert record["often_capitalized_sha256"] == []
    assert record["items_sha256"] == hashlib.sha256(ITEMS.encode()).hexdigest()
    assert "maze_distractors_commit" in record
    assert set(record["word_lists_sha256"]) == {
        "curated_word_list.txt",
        "exclude.txt",
        "often_capitalized.txt",
        "first_names.txt",
        "abbreviations.txt",
        "noun_phrase_breakers.txt",
    }
    assert record["exclude_sha256"] == [hashlib.sha256(b"garden\n").hexdigest()]
    assert record["positions"] == 12
    assert record["positions_short_of_threshold"] == sum(
        row["met"] == "False" for row in reported
    )
    assert record["positions_without_distractor"] == 0


def test_the_format_option_selects_the_writer(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    output = tmp_path / "items.txt"
    run(
        items, output, "--format", "ibex", "--model", model_dir,
        "--device", "cpu", "--min-abs", 20,
    )  # fmt: skip
    assert output.read_text().startswith('[["sub_rel", \'3\'], "Maze", {s:"The')


def test_bad_input_is_one_line_on_stderr_not_a_traceback(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text("type,item_num,sentence,labels\na,1,The dog barked.,x y\n")
    result = run(items, tmp_path / "out.csv", "--model", model_dir, exit_code=1)
    assert (
        result.stderr == "Error: 2 labels for 3 words in 'The dog barked.'.\n"
    )
    assert not (tmp_path / "out.csv").exists()


def test_an_output_at_the_input_path_is_refused_before_anything_runs(
    tmp_path,
):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    # A model that does not exist: the check comes before it is loaded.
    missing_model = tmp_path / "no-model"
    for arguments in (
        [items, tmp_path / "." / "items.csv"],
        [items, tmp_path / "out.csv", "--report", items],
        [items, tmp_path / "out.csv", "--report", tmp_path / "out.csv"],
    ):
        result = run(*arguments, "--model", missing_model, exit_code=1)
        assert result.stderr.startswith("Error: ")
        assert result.stderr.count("\n") == 1
    assert items.read_text() == ITEMS
    assert not (tmp_path / "out.csv").exists()


def test_an_output_path_under_a_file_is_one_line_too(tmp_path):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    (tmp_path / "results").write_text("")
    result = run(
        items, tmp_path / "results" / "out.csv", "--model", tmp_path / "none",
        exit_code=1,
    )  # fmt: skip
    assert result.stderr.startswith("Error: ")
    assert result.stderr.count("\n") == 1


def test_no_frequency_limit_is_recorded_as_valid_json(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    result = run(
        items, tmp_path / "out.csv", "--model", model_dir, "--device", "cpu",
        "--max-frequency-ratio", "inf",
    )  # fmt: skip

    def refuse(constant):
        raise AssertionError(f"{constant} is not JSON")

    record = json.loads(result.stdout, parse_constant=refuse)
    assert record["settings"]["max_frequency_ratio"] == "inf"
    result = run(
        items, tmp_path / "out.csv", "--model", model_dir, "--device", "cpu",
        "--min-abs", "-inf",
    )  # fmt: skip
    record = json.loads(result.stdout, parse_constant=refuse)
    assert record["settings"]["min_abs"] == "-inf"
    result = run(items, tmp_path / "out.csv", "--min-abs", "nan", exit_code=1)
    assert "nan" in result.stderr


def test_a_symlinked_output_is_written_through(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    target = tmp_path / "ibex" / "items.txt"
    target.parent.mkdir()
    target.write_text("old distractors\n")
    link = tmp_path / "items.txt"
    link.symlink_to(target)
    run(
        items, link, "--format", "ibex", "--model", model_dir, "--device", "cpu"
    )
    assert link.is_symlink()
    assert "x-x-x" in target.read_text()
    assert not list(target.parent.glob(".*.part"))


def test_an_unknown_language_is_one_line_too(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    result = run(
        items, tmp_path / "out.csv", "--model", model_dir, "--language", "xx",
        exit_code=1,
    )  # fmt: skip
    assert result.stderr == (
        "Error: wordfreq has no frequencies for the language 'xx'.\n"
    )


def test_an_excluded_word_is_never_a_distractor(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    include = tmp_path / "include.txt"
    include.write_text("garden\n")
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("garden\n")
    common = [items, tmp_path / "out.csv", "--model", model_dir]
    common += ["--device", "cpu", "--min-abs", 0, "--include", include]
    run(*common)
    assert "garden" in (tmp_path / "out.csv").read_text().lower()
    run(*common, "--exclude", exclude)
    assert "garden" not in (tmp_path / "out.csv").read_text().lower()


def test_an_output_at_a_word_list_path_is_refused(tmp_path):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    words = tmp_path / "words.txt"
    words.write_text("garden\n")
    for option in ("--include", "--exclude", "--often-capitalized"):
        result = run(
            items, words, option, words, "--model", tmp_path / "none",
            exit_code=1,
        )  # fmt: skip
        assert "is an input file" in result.stderr
    assert words.read_text() == "garden\n"


def test_the_example_input_runs(tmp_path, model_dir):
    example = Path(__file__).parent.parent / "examples" / "input.csv"
    output = tmp_path / "items.txt"
    run(example, output, "--format", "ibex", "--model", model_dir)
    assert output.read_text().count("x-x-x") == 5


def test_the_defaults_are_the_documented_ones():
    assert vars(Settings()) == {
        "min_delta": 10.0,
        "min_abs": 25.0,
        "max_repeat": 1,
        "max_frequency_ratio": 10.0,
        "min_candidates": 10,
        "break_noun_phrases": True,
        "seed": 0,
    }


def test_a_position_with_no_word_to_try_is_counted_and_warned(
    tmp_path, model_dir, caplog
):
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    include = tmp_path / "include.txt"
    # The one word is a word of the item, so no position may use it.
    include.write_text("cat\n")
    result = run(
        items, tmp_path / "out.csv", "--model", model_dir, "--device", "cpu",
        "--include", include,
    )  # fmt: skip
    record = json.loads(result.stdout)
    assert record["positions_without_distractor"] == 12
    # The two counts partition the failures, as the warnings do.
    assert record["positions_short_of_threshold"] == 0
    assert "12 position(s) have no distractor" in caplog.text
    out = (tmp_path / "out.csv").read_text()
    assert out.count("x-x-x") == 2  # the first words only
    assert out.count("NO-DISTRACTOR") == 12


def test_an_amaze_input_gets_amazes_csv_back(tmp_path, model_dir):
    items = tmp_path / "items.csv"
    items.write_text(
        "sub_rel;3;The cat who the dog scared hid in a box.\n"
        "obj_rel;3;The dog who scared the cat sniffed around the couch.\n"
    )
    output = tmp_path / "out.csv"
    run(items, output, "--model", model_dir, "--device", "cpu", "--min-abs", 20)
    rows = output.read_text().splitlines()
    assert len(rows) == 2
    assert rows[0].startswith("sub_rel;3;The cat who")
    assert rows[0].split(";")[3].startswith("x-x-x ")
    ibex = tmp_path / "out.txt"
    run(
        items,
        ibex,
        "--format",
        "ibex",
        "--model",
        model_dir,
        "--device",
        "cpu",
        "--min-abs",
        20,
    )
    assert ibex.read_text().startswith(
        '[["sub_rel", \'3\'], "Maze", {s:"The cat who'
    )


def test_jspsych_output_is_a_module_and_model_options_reach_the_loader(
    tmp_path, model_dir, monkeypatch
):
    seen = {}
    real = Scorer.from_pretrained

    def spy(model, **kwargs):
        seen.update(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(Scorer, "from_pretrained", staticmethod(spy))
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    output = tmp_path / "stimuli.js"
    run(
        items, output, "--format", "jspsych", "--model", model_dir,
        "--device", "cpu", "--min-abs", 20, "--revision", "main",
        "--bos-token", "<|endoftext|>", "--no-break-noun-phrases",
    )  # fmt: skip
    assert output.read_text().startswith("export const stimuli = [")
    assert seen["revision"] == "main"
    assert seen["bos_token"] == "<|endoftext|>"
    assert seen["device"] == "cpu"


class _Installed:
    def __init__(self, direct_url):
        self.direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return self.direct_url


@pytest.mark.parametrize(
    ("direct_url", "commit"),
    [
        (
            '{"url": "https://github.com/amanirmk/maze-distractor-generator",'
            ' "vcs_info": {"vcs": "git", "commit_id": "abc123"}}',
            "abc123",
        ),
        (
            '{"url": "file:///home/me/clone", "dir_info": {"editable": true}}',
            None,
        ),
        ('{"url": "file:///home/me/clone", "dir_info": {}}', None),
        (None, None),
    ],
)
def test_the_commit_is_the_one_a_git_install_recorded(
    monkeypatch, direct_url, commit
):
    monkeypatch.setattr(
        cli.metadata, "distribution", lambda _name: _Installed(direct_url)
    )
    assert cli._code_commit() == commit


def test_a_list_of_ones_own_joins_the_capitalized_check(
    tmp_path, model_dir, monkeypatch
):
    seen = {}
    real = cli.Vocabulary.load

    def spy(*args, **kwargs):
        vocabulary = real(*args, **kwargs)
        seen["often_capitalized"] = vocabulary.often_capitalized
        return vocabulary

    monkeypatch.setattr(cli.Vocabulary, "load", spy)
    items = tmp_path / "items.csv"
    items.write_text(ITEMS)
    mine = tmp_path / "mine.txt"
    mine.write_text("garden\n")
    result = run(
        items, tmp_path / "out.csv", "--model", model_dir, "--device", "cpu",
        "--often-capitalized", mine,
    )  # fmt: skip
    assert "garden" in seen["often_capitalized"]
    record = json.loads(result.stdout)
    assert record["often_capitalized"] == [str(mine)]
    assert record["often_capitalized_sha256"] == [
        hashlib.sha256(b"garden\n").hexdigest()
    ]
