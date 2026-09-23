"""Rebuild src/maze_distractors/data/noun_phrase_breakers.txt, the words
that cannot continue a noun phrase and so are the only distractors allowed
straight after a determiner.

    uv run --group lists python scripts/build_noun_phrase_breakers.py

Every lowercase word wordfreq knows is judged, not only the curated list,
so that the rule still has words to offer under --include.

After "the", a noun, an adjective, a participle or an adverb can all go on
to make a sentence ("The alluded...", "The promptly..."), however unlikely
the word. A word is listed if it is one of CLOSED_CLASS, or if

- WordNet gives it no noun, adjective or adverb sense,
- it is no participle: not a VBN or VBG form of any verb, and not a past
  form that is also the participle ("alluded", but not "gave"),
- it is a finite form of a WordNet verb that lemminflect also arrives at
  from the verb (WordNet predates "selfie" and "burnout", so a word it
  has never heard of proves nothing; and its suffix-stripping alone would
  take "gived" and "sayes" for verbs), and
- it is not one of NOUNS_SINCE_WORDNET.

What is left is finite verbs, pronouns, determiners, prepositions and
conjunctions.
WordNet's rare senses ("a have", "the go") make the rule strict; it only
ever needs enough words, not every word.
"""

import hashlib
import tempfile
from pathlib import Path

import nltk
import wordfreq
from lemminflect import getAllLemmas, getInflection
from nltk.corpus import wordnet

from maze_distractors.vocabulary import Vocabulary

LIST_FILE = (
    Path(__file__).parents[1]
    / "src/maze_distractors/data/noun_phrase_breakers.txt"
)
# Pinned, so that a rebuild differs from the shipped list only through a
# change to the rule (lemminflect's version is fixed by uv.lock).
WORDNET_SHA256 = (
    "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
)
# Pronouns, determiners, possessives, prepositions, conjunctions and
# wh-words with no use as a noun, adjective or adverb. A determiner cannot
# follow another; the item's own words are kept out elsewhere, so "the"
# is never offered beside "the".
CLOSED_CLASS = frozenset(
    """
    a against albeit although amid among an and any anybody anyone
    anything because beside cannot during each else every everybody
    everyone everything for from hers herself him himself his how if into
    its itself my myself nor of oneself onto our ours ourselves per she
    since some something than that the their theirs them themselves these
    they this those to toward towards unless until unto upon versus via we
    what when whenever where whereas whereby wherein whether which
    whichever whilst whoever whom whomever whose with without you your
    yours yourself yourselves
    """.split()  # noqa: SIM905 -- ninety quoted strings read worse
)
# Verbs with a noun use WordNet (2006) lacks, found by reading every listed
# word of the curated vocabulary. Words outside it have not been read.
NOUNS_SINCE_WORDNET = frozenset(
    """
    alter commit commits download downloads edit edits fail fails install
    inter kindle mainline offs overdose podcast podcasts rebuild recharge
    redesign reload restart reuse revamp reveal reveals sync upload uploads
    """.split()  # noqa: SIM905
)


def verb_lemmas(word: str) -> set[str]:
    """The WordNet verbs ``word`` may be a form of, by either library's
    account."""
    lemmas = set(getAllLemmas(word, upos="VERB").get("VERB", ()))
    lemmas.add(wordnet.morphy(word, wordnet.VERB) or word)
    return {lemma for lemma in lemmas if wordnet.synsets(lemma, wordnet.VERB)}


def inflections(lemma: str, *tags: str) -> set[str]:
    return {form for tag in tags for form in getInflection(lemma, tag)}


def is_participle(word: str) -> bool:
    for lemma in verb_lemmas(word):
        if word in inflections(lemma, "VBN", "VBG"):
            return True
        # Where any past form of a verb is also its participle, so are the
        # rest, whichever lemminflect lists as which ("leaned", "leant");
        # and a regular past always is, even beside an irregular participle
        # ("showed", "shown").
        past = inflections(lemma, "VBD")
        if word in past and past & inflections(lemma, "VBN"):
            return True
        if word != lemma and word.endswith("ed"):
            return True
    return False


def is_finite_verb(word: str) -> bool:
    return any(
        word == lemma or word in inflections(lemma, "VBZ", "VBD")
        for lemma in verb_lemmas(word)
    )


def likely_breaks_noun_phrase(word: str) -> bool:
    # WordNet's senses are for open-class words; "a" the letter and "no"
    # the noun do not make "the a" or "the no" a possible phrase.
    if word in CLOSED_CLASS:
        return True
    nominal = any(
        wordnet.synsets(word, pos=pos)
        for pos in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV)
    )
    if nominal or is_participle(word) or word in NOUNS_SINCE_WORDNET:
        return False
    return is_finite_verb(word)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        if not nltk.download("wordnet", download_dir=directory, quiet=True):
            raise SystemExit("WordNet could not be downloaded.")
        archive = Path(directory) / "corpora" / "wordnet.zip"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != WORDNET_SHA256:
            raise SystemExit(
                "NLTK's WordNet is not the one the list was built from."
            )
        nltk.data.path.insert(0, directory)
        words = sorted(Vocabulary(wordfreq.get_frequency_dict("en")).words)
        listed = [word for word in words if likely_breaks_noun_phrase(word)]
    LIST_FILE.write_text("\n".join(listed) + "\n", encoding="utf-8")
    print(f"{len(listed)} of {len(words)} words written to {LIST_FILE}")


if __name__ == "__main__":
    main()
