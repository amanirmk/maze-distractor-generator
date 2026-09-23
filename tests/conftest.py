"""A tiny random GPT-2 with a byte-level BPE tokenizer trained on the spot:
the real scoring path, with no download and no model worth the name."""

import math

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from maze_distractors.surprisal import Scorer

CORPUS = [
    "The dog chased the cat around the house.",
    "The cat who the dog scared hid in a box.",
    "The dog who scared the cat sniffed around the couch.",
    "A bird sang in the tree while the children played outside.",
    "She remembered that his password expired after many failed attempts.",
] * 20
END = "<|endoftext|>"


def train_tokenizer(*, split_on_spaces: bool = True) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    if split_on_spaces:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=330,
        special_tokens=[END],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    return tokenizer


def build_model(vocab_size: int) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=vocab_size,
        bos_token_id=0,
        eos_token_id=0,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        # Wide enough that token surprisals spread over several bits.
        initializer_range=0.4,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=train_tokenizer(), bos_token=END, eos_token=END
    )


@pytest.fixture(scope="session")
def scorer(tokenizer) -> Scorer:
    return Scorer(build_model(len(tokenizer)), tokenizer)


def reference_surprisal(scorer: Scorer, context: str, word: str) -> float:
    """Surprisal of ``word`` after ``context`` the slow, obvious way: one
    unpadded forward pass per token of the word."""
    encode = scorer.tokenizer.encode
    context_ids = encode(context, add_special_tokens=False)
    token_ids = encode(f"{context} {word}", add_special_tokens=False)
    assert token_ids[: len(context_ids)] == context_ids
    total = 0.0
    for k in range(len(context_ids), len(token_ids)):
        inputs = torch.tensor([[scorer.start_token_id, *token_ids[:k]]])
        with torch.inference_mode():
            logits = scorer.model(input_ids=inputs).logits[0, -1]
        total -= torch.log_softmax(logits, dim=-1)[token_ids[k]].item()
    return total / math.log(2)
