#!/usr/bin/env bash

export device_to_use=${1}
export task_to_run=${task_to_run:-"custom-flores_en-es"}

[ $# -lt 1 ] && echo "Too few arguments! Must give device" && exit

run_flores () {
    model=$1
    gpu=$2
    log_output_folder=$3
    seed_from=${4:-1}
    seed_to=${5:-5}

    model_for_log=$(echo $model | sed 's/\//_/')

    echo $task_to_run
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

        #python -m pudb $(which lm_eval) \
        lm_eval \
            --model hf \
            --device ${gpu} \
            --model_args "pretrained=${model},trust_remote_code=True" \
            --gen_kwargs 'do_sample=True,temperature=1.0' \
            --tasks ${task_to_run} \
            --batch_size 256 \
            --limit 10 \
            --log_samples \
            --output ${seed_output_folder} \
            --verbosity DEBUG \
            --seed "${seed},${seed},${seed},${seed}"
    done
}

export -f run_flores

run_flores 'CohereForAI/aya-expanse-8b' ${device_to_use} ./results/ayadebug-mt-results-and-outputs-apr2024 1 1
