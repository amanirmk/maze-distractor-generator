"""The list-building scripts' options. Their rules run over the whole
vocabulary with the real model, so only what decides where a list is
written is tested here."""

import importlib.util
from pathlib import Path

import pytest
import typer

SCRIPTS = Path(__file__).parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_users_word_list_never_replaces_the_shipped_proper_nouns(tmp_path):
    build = load("build_often_capitalized_list")
    words = tmp_path / "mine.txt"
    assert build._output_for(None, None) == build.LIST_FILE
    for out in (None, build.LIST_FILE, words):
        with pytest.raises(typer.BadParameter):
            build._output_for(words, out)
    assert build._output_for(words, tmp_path / "names.txt") == (
        tmp_path / "names.txt"
    )
