#!/usr/bin/env python3

import argparse
from main import run_stuff

parser = argparse.ArgumentParser(description='Auto-generate Maze materials')

parser.add_argument('input', type=str,
                    help='input file')
parser.add_argument('output', type=str, nargs='?', default=None,
                    help='output file (sentence-level CSV); optional in rejection mode')
parser.add_argument('-p', '--parameters', type=str, default=None,
                    help='parameters file')
parser.add_argument('--format', choices=["ibex", "delim"], default="delim",
                    help='output format, either delimited or for ibex maze')
parser.add_argument('--longform', type=str, default=None, metavar='FILE',
                    help='also write longform output (one row per word position) to FILE')
parser.add_argument('--rejection-file', type=str, default=None, metavar='FILE',
                    help='longform CSV with a "rejected" column; regenerate only marked positions')
parser.add_argument('--num-options', type=int, default=1, metavar='N',
                    help='number of candidate options to generate per rejected position (default: 1)')
args = parser.parse_args()

run_stuff(
    args.input,
    args.output or "",
    parameters=args.parameters,
    outformat=args.format,
    longform_outfile=args.longform,
    rejection_file=args.rejection_file,
    num_options=args.num_options,
)
