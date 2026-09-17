import csv
import logging
import ast

DEFAULTS = {
    "min_delta": 10,
    "min_abs": 25,
    "num_to_test": 100,
    "dictionary_loc": "wordfreq_distractor",
    "dictionary_class": "wordfreq_English_dict",
    "threshold_loc": "wordfreq_distractor",
    "threshold_name": "get_thresholds",
    "model": "gpt2",
    "backend": "transformers",
    "max_repeat": 0,
}


def _read_params_file(file):
    """Parse a colon-delimited params file into a dict."""
    params = {}
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=":", quotechar='"')
        for row in reader:
            if row and not row[0].startswith("#"):
                params[row[0].strip()] = ast.literal_eval(row[1].strip())
    return params


def set_params(source=None):
    """Build a complete parameter dictionary from a file path, a dict, or defaults.

    Args:
        source: One of:
            - str: path to a colon-delimited params file
            - dict: parameter overrides
            - None: use all defaults

    Returns:
        dict with all parameters filled in (user values override defaults).
    """
    if source is None:
        user_params = {}
    elif isinstance(source, dict):
        user_params = dict(source)
    else:
        user_params = _read_params_file(source)

    params = {}
    for key, default in DEFAULTS.items():
        if key in user_params:
            params[key] = user_params.pop(key)
        else:
            logging.info("Using default %s = %s", key, default)
            params[key] = default

    params.update(user_params)
    return params
