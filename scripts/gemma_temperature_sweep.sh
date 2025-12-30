#!/usr/bin/env bash

temperatures=$@
max_concurrent=${max_concurrent:-128}
njobs=${njobs:-3}

run_requests () {
    
    local t=$1

    echo "Running Gemma with temperature t=${t}" > /dev/stderr

    CUDA_VISIBLE_DEVICES="" \
        max_concurrent=${max_concurrent} \
        local_server=yes \
        interactive_mode=no \
        corpus=all \
        lang_pair=all \
        do_sample=yes \
        results_folder="./results/saminmt-llm/gemma_temp_sweep_working/temp${t}/" \
        temperature=${t} \
        bash translate-sami-nmt.sh "google/gemma-3-27b-it"

}

export -f run_requests

parallel --jobs ${njobs} --tag --ungroup 'run_requests {1}' ::: ${temperatures}
