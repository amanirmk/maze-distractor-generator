import logging
import time
import csv
import os
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal, load_surprisal_model
from input import read_input
from output import save_delim, save_json, save_distractor_summary, save_longform, save_updated_longform
from utils import strip_punct

SURPRISAL_LOG_HEADER = ["record_type", "sentence_set_id", "label", "prefix", "word", "surprisal_target", "actual_surprisal", "met_threshold"]


def read_rejection_file(filepath):
    """Read a longform review CSV and return locked and rejected position info.

    Returns:
        locked_by_item: dict mapping item_num (str) -> {label (int): distractor_word (str with punct)}
            for positions not marked as rejected.
        rejected_items: set of item_num strings that have at least one rejected position.
    """
    locked_by_item = {}
    rejected_items = set()
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_num = row["item_num"]
            label = int(row["label"])
            distractor = row["distractor"]
            rejected = row.get("rejected", "").strip()
            if rejected:
                rejected_items.add(item_num)
            else:
                if item_num not in locked_by_item:
                    locked_by_item[item_num] = {}
                locked_by_item[item_num][label] = distractor
    return locked_by_item, rejected_items


def run_stuff(infile, outfile, logfile=None, summaryfile=None, parameters="params.txt", outformat="delim", module_name="stimuli",
              model_override=None, backend_override=None,
              rejection_file=None, num_options=1, longform_outfile=None):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat
    
    Args:
        infile: Input CSV file with sentences
        outfile: Output file path
        logfile: Optional CSV file to log distractor candidates with their surprisals
        parameters: A params file path, a dict of parameters, or None for all defaults
        outformat: Output format ('delim' or 'json')
        module_name: Name for JSON export (if outformat='json')
        model_override: If set, use this model name instead of the one in params
        backend_override: If set, use this backend instead of the one in params
    """
    if outformat not in ["delim", "json"]:
        raise ValueError("outfile format not understood: " + outformat)
    params = set_params(parameters)
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params["dictionary_loc"]),
                         params["dictionary_class"])
    d = dict_class(params)
    model_name = model_override or params["model"]
    backend_name = backend_override or params["backend"]
    backend = load_surprisal_model(model_name, backend=backend_name)
    threshold_func = getattr(importlib.import_module(params["threshold_loc"]),
                             params["threshold_name"])
    repeats = Repeatcounter(params["max_repeat"])

    # Rejection mode: pre-seed repeat counter and identify which sentences to process
    locked_by_item = {}
    rejected_items = None
    if rejection_file:
        locked_by_item, rejected_items = read_rejection_file(rejection_file)
        for locked in locked_by_item.values():
            for dist_word in locked.values():
                repeats.increment(strip_punct(dist_word).lower())

    log_writer = None
    log_handle = None
    if logfile:
        parent = os.path.dirname(logfile)
        if parent:
            os.makedirs(parent, exist_ok=True)
        log_handle = open(logfile, "w", newline="")
        log_writer = csv.writer(log_handle)
        log_writer.writerow(SURPRISAL_LOG_HEADER)

    try:
        for ss in sents.values():
            if rejected_items is not None and ss.id not in rejected_items:
                continue  # skip sentences with no rejections
            locked = locked_by_item.get(ss.id, {})
            logging.info("Processing sentence_set_id %s", ss.id)
            ss.do_surprisals(backend, log_writer=log_writer)
            ss.make_labels()
            ss.do_distractors(backend, d, threshold_func, params, repeats,
                              locked=locked, num_options=num_options, log_writer=log_writer)
    finally:
        if log_handle:
            log_handle.close()

    if rejection_file:
        # In rejection mode, the primary output is the updated longform review file.
        # Only pass the sentences that were actually processed (have distractors assigned).
        processed_sents = {id: ss for id, ss in sents.items() if id in rejected_items}
        if longform_outfile:
            save_updated_longform(longform_outfile, rejection_file, processed_sents)
        if outfile and outfile not in ("/dev/null", ""):
            # Write just the regenerated sentences in sentence-level format (for inspection)
            if outformat == "json":
                save_json(outfile, sents, module_name)
            else:
                save_delim(outfile, sents)
    else:
        if outformat == "json":
            save_json(outfile, sents, module_name)
        else:
            save_delim(outfile, sents)
        if longform_outfile:
            save_longform(longform_outfile, sents)

    if summaryfile:
        save_distractor_summary(summaryfile, sents)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()

    run_stuff("input/test_in.csv", "output/test_output.csv", logfile="output/verbose.csv", parameters="params.txt", outformat="delim", module_name="STIM")
    logging.info("run_stuff completed in %.2fs", time.perf_counter() - t0)