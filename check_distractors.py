"""Check whether a word is a grammatical continuation of a prefix using an LLM."""
import csv
import json
import logging
import os
from typing import Optional

from pydantic import BaseModel


class GrammaticalityJudgment(BaseModel):
    grammatical: bool


SYSTEM_PROMPT = (
    "You are a linguist. Given a text prefix and a candidate next word, "
    "determine whether the candidate word is a grammatically valid continuation "
    "of the prefix. Judge only grammar, not plausibility or meaning."
)


def is_grammatical_continuation(
    prefix: str,
    word: str,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
) -> bool:
    """Query an LLM to decide if *word* is a grammatical next word after *prefix*.

    Uses liteLLM with constrained JSON output so the model can only
    return ``{"grammatical": true}`` or ``{"grammatical": false}``.

    Args:
        prefix: The sentence fragment so far.
        word: The candidate continuation word.
        model: Any liteLLM model string (e.g. 'gpt-4o-mini', 'ollama/llama3').
        api_base: Optional API base URL for self-hosted providers.

    Returns:
        True if the model judges the word to be a grammatical continuation.
    """
    from litellm import completion

    user_content = f'Prefix: "{prefix}"\nWord: "{word}"'

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=GrammaticalityJudgment,
    )
    if api_base is not None:
        kwargs["api_base"] = api_base

    response = completion(**kwargs)
    result = json.loads(response.choices[0].message.content)
    return GrammaticalityJudgment(**result).grammatical


def benchmark_grammaticality(
    infile: str,
    outfile: str,
    model: str = "anthropic/claude-sonnet-4-20250514",
    api_base: Optional[str] = None,
) -> None:
    """Check grammaticality of every target and distractor word in a save_delim CSV.

    Reads a CSV produced by save_delim (columns: type, item_num, sentence,
    distractors, labels).  For each word position (starting from the second
    word, since the first has no prefix), builds the prefix from the target
    sentence and queries the LLM for both the target word and the distractor.

    Args:
        infile: Path to the save_delim CSV.
        outfile: Path for the results CSV.
        model: liteLLM model string.
        api_base: Optional API base URL.
    """
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(infile, "r", newline="") as fin, \
         open(outfile, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow([
            "type", "item_num", "sentence", "word_position",
            "target_word", "distractor_word",
            "is_target_grammatical", "is_distractor_grammatical",
        ])

        for row in reader:
            sentence = row["sentence"]
            distractors = row["distractors"]
            target_words = sentence.split()
            distractor_words = distractors.split()
            print(row["item_num"])
            for i in range(1, len(target_words)):
                prefix = " ".join(target_words[:i])
                target = target_words[i]
                distractor = distractor_words[i] if i < len(distractor_words) else ""

                logging.info(
                    "item %s pos %d: target=%s distractor=%s",
                    row["item_num"], i, target, distractor,
                )

                target_gram = is_grammatical_continuation(
                    prefix, target, model=model, api_base=api_base,
                )
                distractor_gram = False
                if distractor:
                    distractor_gram = is_grammatical_continuation(
                        prefix, distractor, model=model, api_base=api_base,
                    )
                
                writer.writerow([
                    row["type"], row["item_num"], sentence, i,
                    target, distractor,
                    target_gram, distractor_gram,
                ])

#benchmark_grammaticality("output/bench_ns9_distilbert-base-uncased.csv", "output/bench_ns9_check_distilbert_claude.csv", model="anthropic/claude-sonnet-4-20250514")
benchmark_grammaticality("output/bench_ns9_distilroberta-base.csv", "output/bench_ns9_check_roberta_claude.csv", model="anthropic/claude-sonnet-4-20250514")

