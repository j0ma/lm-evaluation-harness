#!/usr/bin/env bash

run_gsm8k () {
    local model=$1
    local log_output_folder=$2
    local seed_from=${3:-1}
    local seed_to=${4:-5}

    model_for_log=$(echo $model | sed 's/\//_/')

    echo $model
    echo $gpu
    echo $log_output_folder
    echo $seed_from
    echo $seed_to

    # create log_output_folder
    mkdir -p $log_output_folder

    for seed in $(seq ${seed_from} ${seed_to})
    do
        echo "Seed: ${seed}"

        local seed_output_folder=${log_output_folder}/gsm8k/${seed}
        mkdir -p ${seed_output_folder}

            #--device ${gpu} \
            #--device auto \
            #--model_args "pretrained=${model},trust_remote_code=True,parallelize=True" \

        lm_eval \
            --model vllm \
            --model_args "pretrained=${model},tensor_parallel_size=$(echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l),dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=1" \
            --gen_kwargs 'do_sample=True,temperature=1.0' \
            --tasks gsm8k \
            --batch_size auto \
            --log_samples \
            --output ${seed_output_folder} \
            --seed "${seed},${seed},${seed},${seed}" 2>&1 | tee ${seed_output_folder}/${model_for_log}.log
    done
}

export -f run_gsm8k

export _seed_from=${1:-1}
export _seed_to=${2:-1}
export debug_output_folder=./results/DEBUG-VLLM-ntrex-results-and-outputs-may2025 

parallel --ungroup --link --jobs 1 \
    "run_gsm8k {1} ${debug_output_folder} ${_seed_from} ${_seed_to}" \
    ::: 'Unbabel/TowerInstruct-Mistral-7B-v0.2' 'CohereForAI/aya-expanse-8b' 'google/gemma-2-9b'

