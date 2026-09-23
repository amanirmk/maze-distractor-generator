import logging
import math

import pytest
import torch
from tokenizers import processors
from transformers import PreTrainedTokenizerFast

from conftest import END, build_model, reference_surprisal, train_tokenizer
from maze_distractors import surprisal
from maze_distractors.surprisal import (
    Scorer,
    TokenizationError,
    detect_start_token_id,
)

WORDS = ["The", "dog", "chased", "the", "cat,", "around", "the", "HOUSE."]
# One, two, five and six tokens under the test tokenizer.
CANDIDATES = ["dog", "house", "table", "zebra", "the", "password"]


def approx(value):
    return pytest.approx(value, abs=1e-3)


def wrapped_tokenizer(template: str, **special) -> PreTrainedTokenizerFast:
    tokenizer = train_tokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=template, special_tokens=[(END, 0)]
    )
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special)


# --- sentences --------------------------------------------------------------


def test_every_word_after_the_first_matches_the_reference(scorer):
    scored = scorer.score_words(WORDS)
    assert len(scored) == len(WORDS) - 1
    for i, word in enumerate(scored, start=1):
        context = " ".join(WORDS[:i])
        assert word.context.text == context
        assert word.surprisal == approx(
            reference_surprisal(scorer, context, WORDS[i])
        )


def test_a_word_that_merges_into_the_text_before_it_is_refused():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=train_tokenizer(split_on_spaces=False),
        bos_token=END,
    )
    # This tokenizer learned "The " and "The dog " as units, so adding a
    # word rewrites the tokens before it.
    merging = Scorer(build_model(len(tokenizer)), tokenizer)
    with pytest.raises(TokenizationError, match="'dog'"):
        merging.score_words(["The", "dog", "chased"])


def test_a_sentence_longer_than_the_model_reads_is_refused(scorer):
    with pytest.raises(ValueError, match="longer than the model reads"):
        scorer.score_words(["dog"] * 64)


# --- candidates -------------------------------------------------------------


def test_exact_candidate_surprisals_match_the_reference(scorer):
    # Contexts of different lengths, so rows of a batch are padded.
    contexts = [word.context for word in scorer.score_words(WORDS)]
    contexts = [contexts[0], contexts[3], contexts[6]]
    scores = scorer.score_candidates(
        contexts, CANDIDATES, [math.inf] * len(contexts)
    )
    for candidate, row in zip(CANDIDATES, scores, strict=True):
        for context, score in zip(contexts, row, strict=True):
            assert score == approx(
                reference_surprisal(scorer, context.text, candidate)
            )


def test_a_value_below_the_bound_is_exact_and_one_above_is_a_lower_bound(
    scorer,
):
    context = scorer.score_words(WORDS)[2].context
    exact = [reference_surprisal(scorer, context.text, c) for c in CANDIDATES]
    # Strictly between two exact values: a bound equal to one of them would
    # let float32 rounding, which differs by platform, decide its side.
    middle = len(exact) // 2
    bound = (sorted(exact)[middle - 1] + sorted(exact)[middle]) / 2
    (scores,) = zip(
        *scorer.score_candidates([context], CANDIDATES, [bound]),
        strict=True,
    )
    assert any(e < bound for e in exact)
    assert any(e >= bound for e in exact)
    for score, truth in zip(scores, exact, strict=True):
        if truth < bound:
            assert score == approx(truth)
        else:
            assert bound <= score <= truth + 1e-3


def test_no_forward_pass_when_first_tokens_already_decide(scorer, monkeypatch):
    context = scorer.score_words(WORDS)[2].context

    def fail(_):
        raise AssertionError("ran the model")

    monkeypatch.setattr(scorer, "predict", fail)
    scorer.score_candidates([context], CANDIDATES, [0.0])


def test_small_batches_give_the_same_surprisals(scorer, monkeypatch):
    contexts = [word.context for word in scorer.score_words(WORDS)][:4]
    bounds = [math.inf] * len(contexts)
    whole = scorer.score_candidates(contexts, CANDIDATES, bounds)
    monkeypatch.setattr(surprisal, "_LOGIT_BUDGET", 1)  # one row a batch
    split = scorer.score_candidates(contexts, CANDIDATES, bounds)
    for a, b in zip(whole, split, strict=True):
        assert a == approx(b)


def test_a_candidate_is_tokenized_in_place_not_alone(scorer):
    # " dog" is one token; "dog" alone is three. Scoring the latter would
    # be the surprisal of text the model never sees mid-sentence.
    context = scorer.score_words(WORDS)[0].context
    ((score,),) = scorer.score_candidates([context], ["dog"], [math.inf])
    first_token = scorer.tokenizer.encode(" dog")
    assert len(first_token) == 1
    assert len(scorer.tokenizer.encode("dog")) > 1
    assert score == approx(context.next_token_surprisals[first_token[0]].item())


# --- the start token --------------------------------------------------------


def test_sequences_start_with_exactly_one_start_token(monkeypatch):
    tokenizer = wrapped_tokenizer(f"{END} $A", bos_token=END)
    adding = Scorer(build_model(len(tokenizer)), tokenizer)
    seen = []
    forward = adding.model.forward

    def spy(**kwargs):
        seen.extend(kwargs["input_ids"].tolist())
        return forward(**kwargs)

    monkeypatch.setattr(adding.model, "forward", spy)
    adding.score_words(["The", "dog"])
    assert seen == [[0, *tokenizer.encode("The dog", add_special_tokens=False)]]


def test_a_tokenizer_that_adds_nothing_gets_its_bos_token(tokenizer):
    assert tokenizer.encode("a") == tokenizer.encode(
        "a", add_special_tokens=False
    )
    assert detect_start_token_id(tokenizer) == tokenizer.bos_token_id == 0


def test_a_tokenizer_that_prepends_a_token_has_named_it():
    # bos_token deliberately unset: the behaviour is what is read.
    tokenizer = wrapped_tokenizer(f"{END} $A")
    assert tokenizer.bos_token_id is None
    assert detect_start_token_id(tokenizer) == 0


def test_without_a_bos_token_the_eos_token_starts_and_says_so(caplog):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=train_tokenizer(), eos_token=END
    )
    with caplog.at_level(logging.WARNING):
        assert detect_start_token_id(tokenizer) == 0
    assert "--bos-token" in caplog.text


def test_a_tokenizer_with_a_bos_token_does_not_warn(tokenizer, caplog):
    with caplog.at_level(logging.WARNING):
        detect_start_token_id(tokenizer)
    assert not caplog.records


def test_no_bos_and_no_eos_is_an_error():
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=train_tokenizer())
    with pytest.raises(ValueError, match="--bos-token"):
        detect_start_token_id(tokenizer)


def test_wrapping_in_more_than_a_leading_token_is_an_error():
    tokenizer = wrapped_tokenizer(f"{END} $A {END}", bos_token=END)
    with pytest.raises(ValueError, match="other than one leading token"):
        detect_start_token_id(tokenizer)


def test_a_named_start_token_overrides_detection():
    tokenizer = wrapped_tokenizer(f"{END} $A {END}", bos_token=END)
    model = build_model(len(tokenizer))
    named = Scorer(model, tokenizer, bos_token="Ġthe")
    assert named.start_token_id == tokenizer.get_vocab()["Ġthe"]
    with pytest.raises(ValueError, match="not in the tokenizer"):
        Scorer(model, tokenizer, bos_token="<nope>")


def test_from_pretrained_forwards_the_revision_and_start_token(
    monkeypatch, tokenizer, scorer
):
    seen = {}

    def load_tokenizer(name, revision=None):
        seen["tokenizer"] = (name, revision)
        return tokenizer

    def load_model(name, revision=None, dtype=None):
        seen["model"] = (name, revision, dtype)
        return scorer.model

    monkeypatch.setattr(
        surprisal.AutoTokenizer, "from_pretrained", load_tokenizer
    )
    monkeypatch.setattr(
        surprisal.AutoModelForCausalLM, "from_pretrained", load_model
    )
    loaded = Scorer.from_pretrained(
        "some/model", revision="abc123", device="cpu", bos_token="a"
    )
    assert seen["tokenizer"] == ("some/model", "abc123")
    assert seen["model"] == ("some/model", "abc123", torch.float32)
    assert loaded.start_token == "a"
