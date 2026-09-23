# maze-distractors

Automatically generate distractors to use in the Maze task (Freedman & Forster,
1985; Forster, Guerrera, & Elliot, 2009).

This repo builds on
[vboyce/maze-distractor-generator](https://github.com/vboyce/maze-distractor-generator),
a modernized version of [A-Maze](https://github.com/vboyce/Maze) (Boyce,
Futrell, & Levy, 2020). It is compatible with causal language models on
HuggingFace and includes additional checks to prevent plausible distractors for
English.

## Quick start

If you already generate distractors with A-Maze, this repo is a drop-in
replacement with the same input and output formats.

```
uvx --from git+https://github.com/amanirmk/maze-distractor-generator maze-distractors input.csv items.txt --format ibex
```

The first run downloads the language model (default: `gpt2-medium`; 1.4 GB).
To test it out, try `examples/input.csv`.
Paste `items.txt` into the `items` array of your Ibex experiment as before.

## Detailed Usage Guide

Clone and `uv sync`, or use `uvx` as above. `maze-distractors --help` lists
every option.

**Input** is a CSV with a header and the columns `type` (can be anything; passed
through to the output), `item_num` (sentences with the same value form an item
and share distractors), `sentence` (words separated by spaces) and, optionally,
`labels` (one label per word). Within an item, words with the same label get the
same distractor; without labels, a word's label is its position. Setting labels
is important for sharing distractors where it matters:

```
type,item_num,sentence,labels
sub_rel,3,The cat who the dog scared hid in a box.,pre_1 pre_2 who art noun verb main_verb post_1 post_2 post_3
obj_rel,3,The dog who scared the cat sniffed around the couch.,pre_1 pre_2 who verb art noun main_verb post_1 post_2 post_3
```

**Output** depends on `--format`: `csv` (the default; the columns `type`,
`item_num`, `sentence`, `distractors`, and `labels`, the last filled in by
position if the input had none), `ibex` (lines for an Ibex Maze experiment) or
`jspsych` (a JavaScript module). Sentences come out in the order they went in.
A sentence's first word has no distractor and is paired with `x-x-x`.
Distractors take on their word's punctuation and capitalization.

**Checking the result.** `--report FILE` writes one row per word: the surprisal
threshold the target word set, the distractor's surprisal, and whether it met the
threshold. Positions that fell short are also logged as warnings. If no word of the
vocabulary could be tried at a position at all (possible with a small
`--include` list), it shows `NO-DISTRACTOR` and issues a warning. This also
prints out a JSON record of the run -- the model and its commit, the wordfreq
version, the settings, and counts of positions short of their threshold or
without a distractor.

**Useful arguments:**

- `--model` takes any causal language model on the Hub (default
  `openai-community/gpt2-medium`); `--revision` pins its commit.
- `--exclude FILE` (repeatable) keeps more words out. `--include FILE` replaces
  the built-in word list with your own.
- `--seed` controls which acceptable distractors are chosen. The same input,
  settings, model, and wordfreq version reproduce the same distractors, up to
  floating point differences between machines.
- `--language CODE` uses wordfreq's frequencies for another language. For
  non-English, please also provide a curated vocab list with `--include`, as
  without it, the pipeline falls back to wordfreq's small wordlist (occurs at
  least once per million).
- Distractor parameters (`--min-delta`, `--min-abs`, `--max-repeat`,
  `--max-frequency-ratio`, `--break-noun-phrases`) are explained below.

## How Distractors are Chosen

Distractors are made implausible by requiring that the language model finds them
unlikely, as measured by surprisal. Typically distractors are also
ungrammatical, and we incorporate heuristics to help prevent surprising but
grammatical distractors.

- Each target word sets a surprisal threshold for its distractor: its own surprisal
  plus `--min-delta` (default 10 bits), and at least `--min-abs` (default 25
  bits). A label shared by several sentences has to meet every one of its
  targets. These defaults carry over from A-Maze, which set them for the
  Gulordava et al. (2018) model; they work well with `gpt2-medium` but may not
  for other models.
- The candidates for a position are the words of the vocabulary within one
  letter of the target word's length and within `--max-frequency-ratio` of its
  frequency (default 10, one order of magnitude). Lengths always reach 4 and
  12, and a word more common than about 160 per million is matched as if just
  that common, so that the shortest, longest, and most common words have
  equals to choose among. If no word matches in length at all, lengths further
  off are tried, with a warning. The candidate list excludes words that are
  used in the item or have been chosen as a distractor `--max-repeat` times
  (default 1, no repeats).
- Straight after a determiner ("the", "a", "his", ...), we try only words that a
  heuristic judges unable to continue a noun phrase: finite verbs, pronouns,
  determiners, prepositions, and conjunctions. After "the", a rare noun is
  surprising only for being rare, and some verb forms ("The alluded...") can
  still go on to make a sentence. `--no-break-noun-phrases` turns this off; it
  is off for languages other than English, with a warning.
- We take the first candidate that meets the threshold in every sentence
  (satisficing, not optimizing). If no candidate does, we take the one that
  falls shortest and report the position.
- A distractor shown capitalized has to meet the threshold both as shown and in
  lower case, since a model can take a capitalized word for a name.

Generating distractors for 30 items in a 2x3 design (240 sentences) takes only
about 70 seconds with `gpt2-medium` on an Apple M4 Pro. It's fast because we
score every target word of a sentence in one forward pass and look up each
candidate's first token in the resulting distribution, which settles most
candidates without running the model again. The result is the same as if we had
scored every candidate in full.

**On the frequency limit.** If we allow the search to stray in frequency, it
drifts toward rarer words, since that is where most of the vocabulary is, and
"pick the more familiar word" becomes a way to do the task. On the 240 sentences
above, removing the limit let only 4 of 441 distractors fall short of their
target, but made 93 of them more than ten times rarer than the target word. With
the limit, 87 fall short instead, by a median of 1.2 bits and never below 19
bits -- still words nobody would choose.

## Vocabulary

Distractors come from A-Maze's curated list of 19k English words (screened for
offensive and sensitive terms), minus its exclusions, both under
`src/maze_distractors/data/`. Frequencies come from
[wordfreq](https://github.com/rspeer/wordfreq). Two further lists are built by
rule rather than by hand, and the scripts under `scripts/` rebuild them:

- `proper_nouns.txt` holds words that are mainly proper nouns ("josh", "oxford",
  "jack"), which in lower case are not the word a reader knows. A word makes the
  list if SUBTLEX-US (Brysbaert & New, 2009) has it capitalized in at least half
  its occurrences and `gpt2-medium` prefers it capitalized mid-sentence by at
  least 3 bits (6 bits, if SUBTLEX-US lacks the word). If you edit the curated
  list, please rerun `uv run python scripts/build_proper_noun_list.py`.
- `noun_phrase_breakers.txt` holds the words we allow straight after a
  determiner, judged from WordNet and lemminflect over every English word
  wordfreq knows. Rebuild with
  `uv run --group lists python scripts/build_noun_phrase_breakers.py`.

No list anticipates everything. Please screen the output distractors for your
own considerations.

## Development

```
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run --group lists ty check src scripts
uv run pytest            # seconds; a tiny random model, no download
uv run pytest -m slow    # the default model, if it is in the local cache
```

The tests compare every surprisal and every choice against a slow reference that
scores one token per forward pass and tries every candidate.

---

**AI Notice:** I (@amanirmk) use coding agents extensively as personal
assistants in my workflow: in writing, coding, and arbitrary related tasks. I
direct all projects, manually review all content, and take full responsibility
for any errors or inaccuracies.
