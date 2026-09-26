# maze-distractors

Automatically generate distractors to use in the Maze task (Freedman & Forster,
1985; Forster, Guerrera, & Elliot, 2009).

This repo builds on
[vboyce/maze-distractor-generator](https://github.com/vboyce/maze-distractor-generator),
a modernized version of [A-Maze](https://github.com/vboyce/Maze) (Boyce,
Futrell, & Levy, 2020). It is intended and engineered for personal use in my
experimental pipeline, but made available to everyone. The repo is compatible
with causal language models on HuggingFace and includes additional checks to
prevent plausible distractors for English. If you publish work using it,
please cite Boyce, Futrell, & Levy (2020) for the general method, along with
the language model you used for scoring.

## Quick start

If you already generate distractors with A-Maze, this repo is a drop-in
replacement with the same input and output formats.

```
uvx --from git+https://github.com/amanirmk/maze-distractor-generator maze-distractors input.csv items.txt --format ibex
```

The first run downloads the language model (default: `gpt2-medium`; 1.4 GB).
To test it out, try
[`examples/input.csv`](https://raw.githubusercontent.com/amanirmk/maze-distractor-generator/main/examples/input.csv).
Paste `items.txt` into the `items` array of your Ibex experiment as before.

## Detailed Usage Guide

Clone and `uv sync`, or use `uvx` as above. `maze-distractors --help` (in a
clone, `uv run maze-distractors --help`) lists every option.

**Input** is a CSV with a header and the columns `type` (can be anything; passed
through to the output), `item_num` (sentences with the same value form an item
and share distractors), `sentence` (words separated by spaces) and, optionally,
`labels` (one label per word; `label` is taken too, and column names are
matched ignoring case and surrounding spaces). Within an item, words with the
same label get the same distractor; without labels, a word's label is its
position. Setting labels is important for sharing distractors where it matters:

```
type,item_num,sentence,labels
obj_rel,3,The cat who the dog scared hid in a box.,pre_1 pre_2 who art noun verb main_verb post_1 post_2 post_3
sub_rel,3,The dog who scared the cat sniffed around the couch.,pre_1 pre_2 who verb art noun main_verb post_1 post_2 post_3
```

For backwards compatibility, the A-Maze input format (no header, columns in
order, semicolon-separated) is also accepted. In rare cases, a file that
A-Maze would have accepted may be refused with an informative message.

**Output** depends on `--format`: `csv` (the default; the columns `type`,
`item_num`, `sentence`, `distractors`, and `labels`, the last filled in by
position if the input had none), `ibex` (lines for an Ibex Maze experiment) or
`jspsych` (a JavaScript module). Sentences come out in the order they went in.
A sentence's first word has no distractor and is paired with `x-x-x`.
Distractors take on their word's punctuation and capitalization.

For backwards compatibility, when given input formatted for A-Maze, the `csv`
output will also follow the A-Maze format.

**Checking the result.** `--report FILE` writes a CSV with one row per word
after the first (which gets no distractor): the target word's and the
distractor's frequencies (per million words), the surprisal threshold the
target word set, the distractor's surprisal, and whether it met the threshold. Positions that
fell short are logged as warnings. If no word of the vocabulary could be tried
at a position at all (possible with a small `--include` list), it shows
`NO-DISTRACTOR` and issues a warning. Every run prints a JSON record to standard
output—the model and its commit, the versions of the scoring libraries, the
settings, and counts of positions short of their threshold or without a
distractor.

**Useful arguments:**

- `--model` takes any causal language model on the Hub (default
  `openai-community/gpt2-medium`); `--revision` pins its commit.
- `--exclude FILE` (repeatable) keeps more words out. `--include FILE` replaces
  the built-in word list with your own; for English, the built-in exclusion
  and often-capitalized lists ([Vocabulary](#vocabulary)) still apply to it.
  `--often-capitalized FILE` (repeatable) adds words to score capitalized too.
- `--seed` controls which acceptable distractors are chosen. The same input,
  settings, model and library versions (all in the run record) reproduce the
  same distractors, up to floating point differences between machines. Choices
  also depend on the input's order: a word chosen for an earlier item is worn
  out for later ones (`--max-repeat`), so excluding one word can move
  positions beyond those that used it. If you require reproducibility, please
  also note that `uvx` resolves to the newest libraries rather than the
  versions in this repo's lockfile.
- `--language CODE` uses wordfreq's frequencies for another language. For
  non-English, please also provide a curated vocab list with `--include`, as
  without it, the pipeline falls back to wordfreq's small wordlist (occurs at
  least once per million).
- Distractor parameters (`--min-delta`, `--min-abs`, `--max-repeat`,
  `--max-frequency-ratio`, `--min-candidates`, `--break-noun-phrases`) are
  explained below.

## How Distractors are Chosen

Distractors are made implausible by requiring that the language model finds them
unlikely, as measured by surprisal. Typically distractors are also
ungrammatical, and we incorporate heuristics to help prevent surprising but
grammatical distractors.

1. We start from the vocabulary—the curated word list or the list you
   provide—minus built-in exclusions when the language is English (see
   [Vocabulary](#vocabulary)). From it we remove every word used in the
   item—as a word or as a distractor—and every word already chosen as a
   distractor in another item `--max-repeat` times (default 1, no repeats).
2. For English, when the target follows a determiner ("the", "a", "his", etc.),
   we only keep words that a heuristic judges unable to continue a noun phrase:
   finite verbs, pronouns, determiners, prepositions, and conjunctions. After
   "the", a rare noun is surprising only for being rare, and some verb forms
   ("The alluded...") can still go on to make a sentence. Use
   `--no-break-noun-phrases` to turn this off.
3. We then only keep words whose frequency is within a factor of
   `--max-frequency-ratio` of the target's (default 10: from 10 times less
   frequent to 10 times more frequent). The range is anchored for extreme
   targets where there would be few candidates. For a target more frequent than
   160 per million, the low end applies the ratio to 160 per million instead of
   to the target. For a target less frequent than every vocabulary word, the
   high end applies the ratio to the least common vocabulary word instead of to
   the target.
4. We next keep only words within one letter of the target's length. Like
   frequency, this window is anchored: targets of 1–2 letters allow words
   up to 4 letters, and targets of 14 or more allow words down to 12. If fewer
   than `--min-candidates` (default 10) are left after all of these steps, we
   widen the length window by a letter on each side until there are enough.
   When one label covers target words of different lengths or frequencies,
   the length window and the frequency range span all of them.
5. Then we sort candidates using a binned distance from the target: first by
   length, one letter per bin, then by log-frequency, one bit per bin. For an
   anchored target, any frequency between the anchor and the target counts as
   zero distance. Within the same bins, the order is fixed by `--seed`.
6. Each target word sets a surprisal threshold for its distractor: its own
   surprisal plus `--min-delta` (default 10 bits), and at least `--min-abs`
   (default 25 bits). The target's surprisal includes any punctuation attached
   to it, so a comma or period raises the bar: a conservative choice kept from
   A-Maze. The default values also carry over from A-Maze, which set them for
   the Gulordava et al. (2018) model; they work well with `gpt2-medium` but may
   not for other models.
7. A distractor must meet the threshold in every context it appears in: all
   positions in all sentences of the item that share its label. When the
   distractor would be capitalized (to match a target word), it must meet the
   threshold both capitalized and in lowercase, whereas A-Maze scores only in
   lowercase. A word that is often capitalized (see [Vocabulary](#vocabulary))
   must meet it capitalized too, even when shown in lowercase, since a reader
   may take "grace" for the name Grace. Both are conservative choices which
   matter most for words that can function as names.
8. We walk through the sorted words and take the first that meets every
   threshold. If none do, we take the one that comes closest and report the
   position. We never widen the tolerated ranges to reach the surprisal
   threshold when there are enough candidates—over-prioritizing any one measure
   tends to produce worse distractors, and the closest candidate is usually
   still near the thresholds.

Generating distractors for 30 items in a 2x2x2 design (240 sentences) takes only
about 70 seconds with `gpt2-medium` on an Apple M4 Pro. It's fast because we
score every target word of a sentence in one forward pass and look up each
candidate distractor's first token in the resulting distribution, which settles
most candidates without running the model again. The result is the same as if we
had scored every candidate in full.

## Vocabulary

Distractors come from A-Maze's curated list of 19k English words, already
screened for offensive and sensitive terms. Frequencies come from
[wordfreq](https://github.com/rspeer/wordfreq). Five more lists sit beside the
curated one under `src/maze_distractors/data/`. The first is taken from A-Maze
and the other four are built via heuristics using the files in `scripts/`.

- `exclude.txt` holds A-Maze's wide-ranging list of words to exclude, which we
  always leave out when the language is set to English. None of them are on the
  curated list, so it only matters for an `--include` list (see
  **Your own list** below).
- `often_capitalized.txt` holds words that are frequently capitalized: names,
  places, titles, brands ("oxford", "professor", "academy"). A word makes the
  list if SUBTLEX-US (Brysbaert & New, 2009) has it capitalized in at least 40%
  of its occurrences and `gpt2-medium` prefers it capitalized mid-sentence by
  at least 3 bits (6 bits, if SUBTLEX-US lacks the word). This list is used to
  enforce checking the surprisal of the capitalized version (step 7 above),
  since (a) the model may find them surprising in lowercase for the case alone
  and (b) capitalized may better reflect how people actually interpret the word.
- `first_names.txt` holds every first name in the 1990 US Census lists that is
  capitalized at least 95% of the time in SUBTLEX-US and occurs in lowercase
  less than once per million words. In contrast to the above list, these are
  clearly names and we exclude them as distractors. Words that are also
  ordinary words, like "sue" and "mark", still survive.
- `abbreviations.txt` holds every USPS state or territory code, as well as
  every abbreviated title in the US Government Publishing Office's
  *Style Manual*, that is capitalized at least 50% of the time in SUBTLEX-US
  and occurs in lowercase less than 10 per million words. These abbreviations
  are also excluded from being used as distractors.
- `noun_phrase_breakers.txt` holds the words we allow straight after a
  determiner (step 2), judged from WordNet and lemminflect over every English
  word wordfreq knows.

Both `first_names.txt` and `abbreviations.txt` can be rebuilt with
`uv run python scripts/build_exclusion_lists.py`. `often_capitalized.txt` is
rebuilt with `uv run python scripts/build_often_capitalized_list.py`, and
`noun_phrase_breakers.txt` is rebuilt with
`uv run --group lists python scripts/build_noun_phrase_breakers.py`.

If you edit the curated word list, please rerun
`uv run python scripts/build_often_capitalized_list.py` as it is created by
checking the words in that list.

**Your own list.** If you use your own word list via `--include`, and the
language is English, our vocabulary checks will still apply. Any words in
`exclude.txt`, `first_names.txt`, or `abbreviations.txt` are removed, and the
words in `often_capitalized.txt` are tested with capitalization. However, the
capitalization list is built only from the curated word list, so any words in
`--include` would need to be checked in a clone using
`uv run python scripts/build_often_capitalized_list.py --words your_list.txt
--out your_capitalized.txt` and then passed to the distractor generator with
`--often-capitalized your_capitalized.txt`.

No list anticipates everything. We recommend always screening the output
distractors for sensitive words and any other considerations.

## Development

```
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run --group lists ty check src scripts
uv run pytest            # seconds; a tiny random model, no download
uv run pytest -m slow    # the default model, if it is in the local cache
```

The tests compare the fast scoring against a slow reference that scores one
token per forward pass, and the search's choices against an exhaustive search
over the same candidates.

---

**AI Notice:** I (@amanirmk) use coding agents extensively as personal
assistants in my workflow: in writing, coding, and arbitrary related tasks. I
direct all projects, manually review all content, and take full responsibility
for any errors or inaccuracies.
