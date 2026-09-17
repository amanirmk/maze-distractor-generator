# maze-distractor-generator

This is an update to Maze distractor generation code, meant to work with a wider variety of more recent models, and to work more easily with my current use cases.

## Input

Input should be a csv with a header row and the following columns

1. "type" contains data to be passed through
2. "item_num" contains an item id -- items with the same id get the same distractors (using labels)
3. "sentence" the sentence
4. (optional) "labels" if there are multiple sentences with the same item_num and they should have distractors matched on something other than word position, use labels to indicate the mappings

Example:
sub_rel,3,The cat who the dog scared hid in a box.,pre_1 pre_2 who art noun verb main_verb post_1 post_2 post_3
obj_rel,3,The dog who scared the cat sniffed around the couch.,pre_1 pre_2 who verb art noun main_verb post_1 post_2 post_3

here the articles of the relative phrases will get the same distractors, as will the verbs, and will the nouns in the relative phrases, even though they are in different positions within the sentences.

## Output

The options for output are a csv (for post-processing to work with anything) and a json file (for use with jspsych).
For csv, it will basically give the input file back, plus a distractor column.
For json, it will return a js module.

## Verbose logging

if you provide a logfile argument to run_stuff then it will log all the surprisals for the target word and potential distractors. Useful for debugging or for having options.

## Calculating surprisal

Surprisal is calculated using some model. Due to potential tokenization issues, we tokenize the prefix + " " + target word and then compare to just the prefix and then take the surprisal of the tokens beyond the prefix. This assumes that exactly how spaces are included in tokens is not relevant (which given that we mostly care about approximate surprisal seems fair).

Currently, we calculate the surprisal of the real words with all punctuation and capitalization included, but we calculate the surprisal of distractors plain (all lower case, no start or end punctuation). I think this is usually a conservative choice (certainly end punctuation will always increase surprisal, start punctuation or capitals could go either way in theory).

## Models that we might want to test

gpt2 (124M)
distilgpt2 (82M)
EleutherAI/gpt-neo-125m (125m)
facebook/opt-125m (125m)
llama3.2-1B (1B)
EleutherAI/pythia-160m
bigscience/bloom-560m
HuggingFaceTB/SmolLM-360M
Qwen/Qwen2-0.5B

## Distractors

curated_word_list has 19.4 K words of 1-14 characters in length (with at least 2\*\*7 tokens / billion words), filtered to be "real" words of all lower case, and excluding offensive words and (at least decreasing) words associated with violence and other sensitive topics. No guarantees.

We recommend doing a level of distractor filtering that matches your use case. Some potential options/tools:

- Use TODO implement to see and screen the list of distractor words (to your own sensibilities about what "counts" as a word or is in good taste); then find where any problem distractors were and replace/regenerate.
- Use TODO implement LLM API calls to assess rejectability (note that LLM meta-linguistic judgments are not necessarily trustworthy)
- Screen all or especially critical items by hand by running through them yourself
- Pilot materials on a few participants and re-generate/fix distractors that multiple pilot participants get wrong.
  For many use cases it might not matter that much if a few plausible distractors get through and so filtering may not be worth it. These are options for if you want more quality control, for instance for a high-stakes experiment or an experiment with ex. children.
