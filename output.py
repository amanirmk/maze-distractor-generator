import csv
import os
from collections import Counter

import json
from utils import strip_punct, copy_punct

LONGFORM_FIELDNAMES = ["type", "item_num", "label", "prefix", "real_word", "distractor", "options", "rejected"]


def save_delim(outfile, all_sentences):
    '''Saves results to a file in semicolon delimited format
    basically same as the original input with another column for distractor sentence
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none
    will write a semicolon delimited file with
    column 1 = "tag"/condition copied over from item_to_info (from input file)
    column 2 = item number
    column 3 = good sentence
    column 4 = string of distractor words in order.
    column 5 = string of labels in order. '''
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)  
    with open(outfile, 'w+', newline="") as f:
        writer=csv.writer(f,delimiter=",")
        writer.writerow(["type", "item_num", "sentence", "distractors", "labels"])
        for sentence_set in all_sentences.values():
            for sentence in sentence_set.sentences:
                writer.writerow([sentence.tag,sentence.id,sentence.word_sentence,sentence.distractor_sentence,sentence.label_sentence])




def save_json(outfile, all_sentences, name="stimuli"):
    '''Saves results as a JavaScript module (for use with jspsych)
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    name: variable name for the exported stimuli list (optional)
    Returns: none
    Writes a .js file with "export const stimuli = [...]" where each item has:
    * item_type (tag/condition)
    * id (item number)
    * sentence (original sentence)
    * distractor (distractor sentence)
    * labels (label string)
    '''
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    items = []
    for sentence_set in all_sentences.values():
        for sentence in sentence_set.sentences:
            items.append({
                "item_type": sentence.tag,
                "id": sentence.id,
                "sentence": sentence.word_sentence,
                "distractor": sentence.distractor_sentence,
                "labels": sentence.label_sentence,
            })
    js_content = f"export const {name} = " + json.dumps(items, indent=2) + ";\n"
    with open(outfile, "w") as f:
        f.write(js_content)


def save_longform(outfile, all_sentences):
    """Save results in longform: one row per non-first-word position across all sentences.

    Columns: type, item_num, label, prefix, real_word, distractor, options, rejected

    'distractor' is the assigned word (with punctuation applied).
    'options' is a comma-separated list of alternative candidates beyond the first
    (populated when num_options > 1 was used; empty otherwise).
    'rejected' is always empty on initial output — the reviewer fills this in.
    """
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONGFORM_FIELDNAMES)
        writer.writeheader()
        for ss in all_sentences.values():
            for sentence in ss.sentences:
                for i in range(1, len(sentence.labels)):
                    lab = sentence.labels[i]
                    real_word = sentence.words[i]
                    distractor = sentence.distractors[i]
                    opts = ss.label_options.get(lab, [])
                    # opts[0] is already reflected in distractor; opts[1:] are alternatives
                    alt_strings = [copy_punct(real_word, o) for o in opts[1:]]
                    writer.writerow({
                        "type": sentence.tag,
                        "item_num": sentence.id,
                        "label": lab,
                        "prefix": " ".join(sentence.words[:i]),
                        "real_word": real_word,
                        "distractor": distractor,
                        "options": ", ".join(alt_strings),
                        "rejected": "",
                    })


def save_updated_longform(outfile, original_longform_path, all_sentences):
    """Update a longform review file with newly generated distractors for rejected positions.

    Reads the original longform file, replaces rejected rows with new distractors from
    all_sentences, clears the 'rejected' flag, and writes everything to outfile.
    Non-rejected rows are written unchanged. The 'options' column is added if not present.

    all_sentences should contain only the sentence sets that were regenerated.
    """
    # Read original, preserving row order
    with open(original_longform_path, newline="") as f:
        reader = csv.DictReader(f)
        orig_fieldnames = reader.fieldnames or []
        orig_rows = list(reader)

    # Build lookup of new distractors: (item_num_str, label_str) -> (distractor, options_str)
    new_data = {}
    for ss in all_sentences.values():
        for sentence in ss.sentences:
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                real_word = sentence.words[i]
                distractor = sentence.distractors[i]
                opts = ss.label_options.get(lab, [])
                alt_strings = [copy_punct(real_word, o) for o in opts[1:]]
                new_data[(str(sentence.id), str(lab))] = (distractor, ", ".join(alt_strings))

    # Determine output fieldnames: preserve original columns, ensure 'options' and 'rejected' present
    out_fieldnames = list(orig_fieldnames)
    if "options" not in out_fieldnames:
        # Insert before 'rejected' if it exists, otherwise append
        if "rejected" in out_fieldnames:
            out_fieldnames.insert(out_fieldnames.index("rejected"), "options")
        else:
            out_fieldnames.append("options")
    if "rejected" not in out_fieldnames:
        out_fieldnames.append("rejected")

    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in orig_rows:
            key = (row["item_num"], row["label"])
            if key in new_data and row.get("rejected", "").strip():
                new_dist, new_opts = new_data[key]
                row["distractor"] = new_dist
                row["options"] = new_opts
                row["rejected"] = ""
            writer.writerow(row)


def save_distractor_summary(outfile, all_sentences):
    """Saves a CSV summarizing distractor usage: each distinct distractor and its count.

    Args:
        outfile: path to the output CSV file
        all_sentences: dictionary of sentence_set objects
    """
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    counts = Counter()
    for sentence_set in all_sentences.values():
        for sentence in sentence_set.sentences:
            for word in sentence.distractors:
                counts[strip_punct(word).lower()] += 1
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distractor", "count"])
        for word in sorted(counts):
            writer.writerow([word, counts[word]])
