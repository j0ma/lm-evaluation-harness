#!/usr/bin/env bash

run_ntrex () {
    model=$1
    log_output_folder=$2
    seed_from=${3:-1}
    seed_to=${4:-5}

    gpu=$(pick-gpu --debug) || exit 1

    echo "Picked GPU: ${gpu}" >/dev/stderr

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

        local seed_output_folder=${log_output_folder}/ntrex/${seed}
        mkdir -p ${seed_output_folder}

            #--device ${gpu} \
            #--device auto \

        lm_eval \
            --model hf \
            --model_args "pretrained=${model},trust_remote_code=True,parallelize=True" \
            --gen_kwargs 'do_sample=True,temperature=1.0' \
            --tasks custom_flores \
            --batch_size 128 \
            --log_samples \
            --output ${seed_output_folder} \
            --seed "${seed},${seed},${seed},${seed}" 2>&1 | tee ${seed_output_folder}/${model_for_log}.log
    done
}

export -f run_ntrex

#CUDA_VISIBLE_DEVICES=3 run_ntrex 'Unbabel/TowerInstruct-Mistral-7B-v0.2' 'cuda:3' ./results/ntrex-results-and-outputs-apr2024 1 5  &
#CUDA_VISIBLE_DEVICES=4 run_ntrex 'CohereForAI/aya-expanse-8b' 'cuda:4' ./results/ntrex-results-and-outputs-apr2024 1 5  &

#wait

#run_ntrex 'Unbabel/TowerInstruct-Mistral-7B-v0.2' ./results/DEBUG-PICK-GPU-ntrex-results-and-outputs-may2025 1 5  &
#run_ntrex 'CohereForAI/aya-expanse-8b' ./results/DEBUG-PICK-GPU-ntrex-results-and-outputs-may2025 1 5  &

export debug_output_folder=./results/DEBUG-PICK-GPU-ntrex-results-and-outputs-may2025 

parallel --ungroup --delay 60 --link --jobs 2 \
    "run_ntrex {1} ${debug_output_folder} 1 1" \
    ::: 'google/gemma-2-9b' #'Unbabel/TowerInstruct-Mistral-7B-v0.2' 'CohereForAI/aya-expanse-8b'
