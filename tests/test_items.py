import pytest

from maze_distractors.items import read_sentences

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
