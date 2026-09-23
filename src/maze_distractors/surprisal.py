"""Word surprisal under a causal language model, shaped for a distractor
search: every word of a sentence from one forward pass, and many candidate
words after the same contexts at once.

Text is always tokenized as it would be in place -- a candidate is encoded
as ``context + " " + candidate``, never alone, because most tokenizers give
a word different tokens at the start of a string than after a space.
"""

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Self

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# Elements in one batch of logits (rows x length x vocabulary): 1 GiB at
# float32, so a 256k-entry vocabulary cannot exhaust memory.
_LOGIT_BUDGET = 2**28


class TokenizationError(ValueError):
    """Appending a word changed the tokens of the text before it, so the
    word has no tokens of its own to score."""


@dataclass(frozen=True)
class Context:
    """The words before a position, as the model sees them."""

    text: str
    token_ids: tuple[int, ...]
    next_token_surprisals: torch.Tensor  # bits, one per vocabulary entry


@dataclass(frozen=True)
class ScoredWord:
    surprisal: float  # bits
    context: Context


def detect_start_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """The token a sequence starts with, read off the tokenizer's own
    behaviour. It is prepended by this module exactly once, to text encoded
    without special tokens.

    A tokenizer that prepends one token when asked for special tokens has
    named it. One that adds nothing (GPT-2, Pythia) leaves its BOS token,
    or failing that (Qwen) its EOS token, which is what separates documents
    in such a model's training data.
    """
    plain = tokenizer.encode("a", add_special_tokens=False)
    wrapped = tokenizer.encode("a", add_special_tokens=True)
    if wrapped == plain:
        if tokenizer.bos_token_id is not None:
            return tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            logger.warning(
                "The tokenizer has no BOS token; starting sequences with its "
                "EOS token %r. Name another with --bos-token if that is not "
                "what the model was trained with.",
                tokenizer.eos_token,
            )
            return tokenizer.eos_token_id
        raise ValueError(
            "The tokenizer has neither a BOS nor an EOS token; name the "
            "start token with --bos-token."
        )
    if wrapped[1:] == plain:
        return wrapped[0]
    raise ValueError(
        f"The tokenizer wraps text in special tokens other than one leading "
        f"token ({plain} became {wrapped}); name the start token with "
        "--bos-token."
    )


def _vocabulary_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    token_id = tokenizer.get_vocab().get(token)
    if token_id is None:
        raise ValueError(f"{token!r} is not in the tokenizer's vocabulary.")
    return token_id


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def token_prefix_is_unchanged(
    prefix: Sequence[int], tokens: Sequence[int]
) -> bool:
    """``tokens`` starts with every token of ``prefix``, so that what was
    appended to the prefix's text has tokens of its own."""
    return list(tokens[: len(prefix)]) == list(prefix)


def to_bits(logits: torch.Tensor) -> torch.Tensor:
    return -torch.log_softmax(logits.float(), dim=-1) / math.log(2)


class Scorer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        bos_token: str | None = None,
    ):
        """``bos_token`` names the token every sequence starts with; when it
        is None the token is worked out from the tokenizer (see
        ``detect_start_token_id``)."""
        self.model = model.eval()
        self.tokenizer = tokenizer
        # A multimodal checkpoint (Gemma 3 4B) keeps its language model's
        # settings one level down.
        self._config = model.config.get_text_config()
        self.start_token_id = (
            detect_start_token_id(tokenizer)
            if bos_token is None
            else _vocabulary_id(tokenizer, bos_token)
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        bos_token: str | None = None,
    ) -> Self:
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        # float32 whatever the checkpoint declares: half precision puts
        # surprisal on a grid coarser than the differences thresholds are
        # compared at.
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.float32
        )
        loaded = next(model.parameters()).dtype
        if loaded != torch.float32:
            raise ValueError(f"{name} loaded as {loaded}, not float32")
        # ty misreads transformers' wrapped .to() and AutoTokenizer's return.
        model.to(device or _default_device())  # ty: ignore[invalid-argument-type]
        return cls(model, tokenizer, bos_token=bos_token)  # ty: ignore[invalid-argument-type]

    @property
    def start_token(self) -> str:
        # One id decodes to one string; the stubs allow for a batch.
        return self.tokenizer.decode([self.start_token_id])  # ty: ignore[invalid-return-type]

    def _encode(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def _tokenize_words(
        self, words: Sequence[str]
    ) -> tuple[list[int], list[int]]:
        """The sentence's tokens, and for each word the number of tokens up
        to and including it -- checked one word at a time, so a word's
        tokens are those the model sees when the sentence is read whole."""
        encoded = self._encode(
            [" ".join(words[: i + 1]) for i in range(len(words))]
        )
        for i, (shorter, longer) in enumerate(pairwise([[], *encoded])):
            if not token_prefix_is_unchanged(shorter, longer) or len(
                longer
            ) == len(shorter):
                raise TokenizationError(
                    f"{words[i]!r} does not tokenize as a separate word in "
                    f"{' '.join(words)!r}."
                )
        return encoded[-1], [len(token_ids) for token_ids in encoded]

    @torch.inference_mode()
    def predict(self, batch: Sequence[Sequence[int]]) -> torch.Tensor:
        """``result[row, position]`` scores every vocabulary entry as
        ``batch[row][position]``, given the start token and
        ``batch[row][:position]``.

        Rows are padded on the right: under causal attention a position
        never sees what follows it, so padding cannot reach a real token.
        """
        width = max(len(token_ids) for token_ids in batch) + 1
        limit = getattr(self._config, "max_position_embeddings", None)
        if limit is not None and width > limit:
            raise ValueError(
                f"A sentence of {width} tokens is longer than the model "
                f"reads ({limit})."
            )
        input_ids = torch.full((len(batch), width), self.start_token_id)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, token_ids in enumerate(batch):
            input_ids[row, 1 : len(token_ids) + 1] = torch.tensor(token_ids)
            attention_mask[row, : len(token_ids) + 1] = 1
        return self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
        ).logits

    def score_words(self, words: Sequence[str]) -> list[ScoredWord]:
        """Score ``words[1:]``, in order. The first word is not scored: a
        Maze sentence opens without a distractor."""
        token_ids, word_ends = self._tokenize_words(words)
        # [position, vocabulary]; on the CPU, where the rows kept as contexts
        # are cheap to index a few candidates at a time.
        surprisals = to_bits(self.predict([token_ids])[0]).cpu()
        per_token = surprisals[
            torch.arange(len(token_ids)), torch.tensor(token_ids)
        ]
        if not torch.isfinite(per_token).all():
            raise ValueError(
                f"The model gives a token of {' '.join(words)!r} zero "
                "probability."
            )
        return [
            ScoredWord(
                surprisal=per_token[start:end].sum().item(),
                context=Context(
                    text=" ".join(words[:i]),
                    token_ids=tuple(token_ids[:start]),
                    next_token_surprisals=surprisals[start],
                ),
            )
            for i, (start, end) in enumerate(pairwise(word_ends), start=1)
        ]

    def context_for(self, text: str) -> Context:
        """The model's state after ``text``, for scoring what may follow."""
        token_ids = self._encode([text])[0]
        surprisals = to_bits(self.predict([token_ids])[0]).cpu()
        return Context(
            text=text,
            token_ids=tuple(token_ids),
            next_token_surprisals=surprisals[len(token_ids)],
        )

    def score_candidates(
        self,
        contexts: Sequence[Context],
        candidates: Sequence[str],
        exact_below: Sequence[float],
    ) -> list[list[float]]:
        """Surprisal of each candidate after each context, in bits, as
        ``result[candidate][context]``.

        A value below that context's ``exact_below`` is exact. A value at or
        above it may be the first token's surprisal alone. That is a lower
        bound on the word's, since every further token adds a non-negative
        amount, so "at least ``exact_below``" holds either way and the
        remaining tokens are never run. Pass ``math.inf`` for exact values.
        """
        result = [[math.nan] * len(contexts) for _ in candidates]
        unresolved: list[tuple[int, int]] = []
        sequences: list[tuple[list[int], int]] = []
        for j, (context, bound) in enumerate(
            zip(contexts, exact_below, strict=True)
        ):
            n_context = len(context.token_ids)
            encoded = self._encode([f"{context.text} {c}" for c in candidates])
            for candidate, token_ids in zip(candidates, encoded, strict=True):
                if (
                    not token_prefix_is_unchanged(context.token_ids, token_ids)
                    or len(token_ids) == n_context
                ):
                    raise TokenizationError(
                        f"{candidate!r} does not tokenize as a separate "
                        f"word after {context.text!r}."
                    )
            first_tokens = torch.tensor([ids[n_context] for ids in encoded])
            first_surprisals = context.next_token_surprisals[first_tokens]
            for k, first in enumerate(first_surprisals.tolist()):
                result[k][j] = first
                if first < bound and len(encoded[k]) > n_context + 1:
                    unresolved.append((k, j))
                    sequences.append((encoded[k], n_context))
        for (k, j), surprisal in zip(
            unresolved, self.score_suffixes(sequences), strict=True
        ):
            result[k][j] = surprisal
        return result

    def score_exactly(
        self, contexts: Sequence[Context], candidates: Sequence[str]
    ) -> list[list[float]]:
        """Surprisal of each candidate after each context, exact throughout:
        ``score_candidates`` with no bound to stop at."""
        return self.score_candidates(
            contexts, candidates, [math.inf] * len(contexts)
        )

    def score_suffixes(
        self, sequences: Sequence[tuple[list[int], int]]
    ) -> list[float]:
        """For each ``(token_ids, n_context)``, the summed surprisal of
        ``token_ids[n_context:]`` given the tokens before them."""
        totals = [math.nan] * len(sequences)
        for batch in self.batch_by_length([len(ids) for ids, _ in sequences]):
            logits = self.predict([sequences[i][0] for i in batch])
            for row, i in enumerate(batch):
                token_ids, n_context = sequences[i]
                suffix = torch.tensor(token_ids[n_context:])
                # Position p of a row predicts token_ids[p].
                predictions = to_bits(logits[row, n_context : len(token_ids)])
                totals[i] = (
                    predictions[torch.arange(len(suffix)), suffix].sum().item()
                )
        return totals

    def batch_by_length(self, lengths: Sequence[int]) -> Iterator[list[int]]:
        """Indices grouped longest first, so each batch pads little and its
        logits stay within _LOGIT_BUDGET."""
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        while order:
            per_row = (lengths[order[0]] + 1) * self._config.vocab_size
            size = max(1, _LOGIT_BUDGET // per_row)
            yield order[:size]
            order = order[size:]
