import pytest

from maze_distractors.items import detect_layout, read_sentences

HEADER = "type,item_num,sentence,labels\n"


def read(tmp_path, text):
    file = tmp_path / "items.csv"
    file.write_text(text)
    return read_sentences(file)


def test_sentences_are_read_in_input_order(tmp_path):
    sentences = read(
        tmp_path,
        "\ufeff"  # a spreadsheet's byte-order mark
        + HEADER
        + "a,2,The dog barked.,\n"
        + 'b,1,"The cat ran, far away. ",x y z w v\n'
        + "c,2,The dog slept.,\n",
    )
    assert [(s.tag, s.item) for s in sentences] == [
        ("a", "2"),
        ("b", "1"),
        ("c", "2"),
    ]
    assert sentences[1].words == ("The", "cat", "ran,", "far", "away.")
    assert sentences[1].labels == ("x", "y", "z", "w", "v")


def test_without_labels_a_word_is_labelled_by_its_position(tmp_path):
    (sentence,) = read(
        tmp_path, "type,item_num,sentence\na,1,The dog barked.\n"
    )
    assert sentence.labels == ("0", "1", "2")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("type,sentence\na,The dog.\n", "item_num"),
        (HEADER + "a,1,The dog barked.,x y\n", "2 labels for 3 words"),
        (HEADER + "a,1,The dog barked.,x y y\n", "repeat"),
        (HEADER + "a,1,,\n", "empty sentence"),
        (HEADER + "a,,The dog barked.,\n", "no item_num"),
        (HEADER + "a,1\n", "Line 2 .* more or fewer cells"),
        (HEADER + "a,1,The dog - barked.,\n", "without a letter or digit"),
        (
            HEADER + "a,1,The dog barked.,x y z\nb,1,Dogs bark loudly.,y x w\n",
            "first word and a later one",
        ),
        (  # a labels cell left blank beside a labelled sibling
            HEADER + "a,1,The dog barked.,x y z\nb,1,Dogs bark loudly.,\n",
            "shares no label",
        ),
    ],
)
def test_malformed_input_is_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read(tmp_path, text)


# --- A-Maze's own file ---------------------------------------------------
#
# Semicolon-separated, no header, the same columns by position: read as it
# is, so A-Maze users need not convert anything.

AMAZE = (
    "ambiguous-nomod-final;00b01270271c67c9;The office worker eventually "
    "remembered his password expired.;standard0 standard1 standard2 standard3 "
    "standard4 standard5 standard6 critical\n"
    "ambiguous-postmod-final;00b01270271c67c9;The office worker eventually "
    "remembered his password to the website expired.;standard0 standard1 "
    "standard2 standard3 standard4 standard5 standard6 postmod0 postmod1 "
    "postmod2 critical\n"
)


def test_amaze_input_is_read_as_it_is(tmp_path):
    file = tmp_path / "items.csv"
    file.write_text(AMAZE)
    assert not detect_layout(file).headed
    first, second = read_sentences(file)
    assert first.tag == "ambiguous-nomod-final"
    assert first.item == second.item == "00b01270271c67c9"
    assert first.words[-1] == "expired."
    assert first.labels[-1] == second.labels[-1] == "critical"
    assert len(second.labels) == len(second.words) == 11


def test_amaze_rows_without_labels_or_with_the_wrong_cells(tmp_path):
    (sentence,) = read(tmp_path, "a;1;The dog barked.\n")
    assert sentence.labels == ("0", "1", "2")
    with pytest.raises(ValueError, match=r"Line 2 .* 5 cells"):
        read(tmp_path, "a;1;The dog barked.\nb;1;The cat sat.;x y z;extra\n")
    with pytest.raises(ValueError, match="neither a header"):
        read(tmp_path, "a,1,The dog barked.\n")


# --- headers as spreadsheets write them ----------------------------------


@pytest.mark.parametrize(
    "header",
    ["type,item_num,sentence,label", "Type, Item_num, Sentence, Labels "],
)
def test_column_names_are_matched_loosely_and_label_is_labels(tmp_path, header):
    (sentence,) = read(tmp_path, header + "\na,1,The dog barked.,x y z\n")
    assert sentence.labels == ("x", "y", "z")


def test_a_semicolon_file_with_a_header_is_not_mistaken_for_amaze(tmp_path):
    file = tmp_path / "items.csv"
    file.write_text(
        "type;item_num;sentence;labels\na;1;The dog barked.;x y z\n"
    )
    assert detect_layout(file).headed
    (sentence,) = read_sentences(file)
    assert sentence.tag == "a"
    assert sentence.labels == ("x", "y", "z")


@pytest.mark.parametrize(
    "header",
    [
        '"type","item_num","sentence","labels"',
        '"type";"item_num";"sentence";"labels"',
    ],
)
def test_quoted_column_names_are_a_header_as_r_writes_them(tmp_path, header):
    # write.csv and write.csv2 quote every name.
    delimiter = header[6]
    row = delimiter.join(["a", "1", "The dog barked.", "x y z"])
    (sentence,) = read(tmp_path, header + "\n" + row + "\n")
    assert sentence.tag == "a"
    assert sentence.labels == ("x", "y", "z")


@pytest.mark.parametrize(
    "header", ["condition;item;sentence;labels", "type,item,sentence"]
)
def test_a_header_with_other_names_is_refused_not_read_as_a_sentence(
    tmp_path, header
):
    delimiter = ";" if ";" in header else ","
    row = delimiter.join(["a", "1", "The dog barked.", "x y z"])
    with pytest.raises(ValueError, match=r"looks like a header.*item_num"):
        read(tmp_path, header + "\n" + row + "\n")


def test_a_sentence_left_open_by_a_quote_mark_is_refused(tmp_path):
    with pytest.raises(ValueError, match="more than one line"):
        read(tmp_path, 'b;1;"The dog barked.\nc;2;The cat sat.\n')
    # With a header the run-on row is already one cell short.
    with pytest.raises(ValueError, match="fewer cells than the header"):
        read(tmp_path, HEADER + 'b,1,"The dog barked.,x y z\nc,2,The cat.,\n')


def test_quote_marks_read_as_csv_quoting_are_pointed_out(tmp_path, caplog):
    rows = [
        'a;1;"Stop the car," he said.',  # by hand: the quotes are lost
        'b;2;"""Stop the car,"" he said."',  # as a spreadsheet writes it
        'c;3;He said "stop" twice.',
    ]
    by_hand, spreadsheet, inside = read(tmp_path, "\n".join(rows) + "\n")
    assert by_hand.words[2] == "car,"
    assert spreadsheet.words[2] == 'car,"'
    assert inside.words[2] == '"stop"'
    assert "line(s) 1:" in caplog.text
    assert "not shown" in caplog.text


def test_amaze_rows_may_end_in_empty_cells(tmp_path):
    (sentence,) = read(tmp_path, "a;1;The dog barked.;x y z;;\n")
    assert sentence.labels == ("x", "y", "z")


def test_unlabelled_sentences_are_pointed_out(tmp_path, caplog):
    read(tmp_path, HEADER + "a,1,The dog barked.,\n")
    assert "1 of 1 sentence(s) have no labels" in caplog.text
    caplog.clear()
    read(tmp_path, HEADER + "a,1,The dog barked.,x y z\n")
    assert "no labels" not in caplog.text
