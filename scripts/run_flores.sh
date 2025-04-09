#!/usr/bin/env bash

run_flores () {
    model=$1
    gpu=$2
    log_output_folder=$3
    seed_from=${4:-1}
    seed_to=${5:-5}

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

        local seed_output_folder=${log_output_folder}/flores/${seed}
        mkdir -p ${seed_output_folder}

        lm_eval \
            --model hf \
            --device ${gpu} \
            --model_args "pretrained=${model},trust_remote_code=True" \
            --gen_kwargs 'do_sample=True,temperature=1.0' \
            --tasks custom_flores \
            --batch_size 256 \
            --log_samples \
            --output ${seed_output_folder} \
            --seed "${seed},${seed},${seed},${seed}" 2>&1 | tee ${seed_output_folder}/${model_for_log}.log
    done
}

export -f run_flores

run_flores 'Unbabel/TowerInstruct-Mistral-7B-v0.2' 'cuda:1' ./results/custom-flores-results-and-outputs-apr2024 1 5  &
run_flores 'CohereForAI/aya-expanse-8b' 'cuda:5' ./results/custom-flores-results-and-outputs-apr2024 1 5  &

wait

#parallel --dry-run --link --jobs 2 \
    #'run_flores {1} {2} ./results/mt-results-and-outputs-apr2024 1 1' \
    #::: 'Unbabel/TowerInstruct-Mistral-7B-v0.2' 'CohereForAI/aya-expanse-8b' \
    #::: 'cuda:0' 'cuda:1'
    
# TODO: ntrex
