#!/usr/bin/env bash

task_name=$1
num_fewshot=${num_fewshot:-0}
num_examples=${num_examples:-10}

output_folder="./written-outputs/saminmt-llm/${task_name}"

mkdir -vp ${output_folder}

python -m scripts.write_out \
    --output_base_path ${output_folder} \
    --tasks ${task_name} \
    --sets test \
    --num_fewshot ${num_fewshot} \
    --num_examples ${num_examples}
