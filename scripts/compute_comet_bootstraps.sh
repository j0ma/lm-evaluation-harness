#!/usr/bin/env bash

export jsonl_file=$1
export bootstrap_iters=${2:-100}

export lang_pair=$(basename $jsonl_file | cut -f2,3 -d-)
export jsonl_output_file=$(dirname $jsonl_file)/bootstrapped-comet-${lang_pair}.jsonl

python scripts/bootstrap_comet.py \
    -b 100 \
    ${jsonl_file} \
    2> /dev/null \
    > ${jsonl_output_file}
