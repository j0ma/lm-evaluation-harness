#!/usr/bin/env bash

set -euo pipefail

# Model URI
model_uri=${1:-$default_model_uri}

if [ -z "$model_uri" ]; then
    echo "Error: No model URI provided. Please provide a model URI as the first argument."
    exit 1
fi

# Optionally split model to multiple GPUs
split_model_multiple_gpus=${split_model_multiple_gpus:-no}

# Optionally enable interactive mode
interactive_mode=${interactive_mode:-no}

if [ "$interactive_mode" == "yes" ]; then
    corpus_options=("all" "uit" "yle")
    lang_pair_options=("all" "fin-sme" "sme-fin")
    echo "Select corpus:"
    select corpus in "${corpus_options[@]}"; do
        if [[ " ${corpus_options[*]} " == *" $corpus "* ]]; then
            break
        else
            echo "Invalid option. Please try again."
        fi
    done

    echo "Select language pair:"
    select lang_pair in "${lang_pair_options[@]}"; do
        if [[ " ${lang_pair_options[*]} " == *" $lang_pair "* ]]; then
            break
        else
            echo "Invalid option. Please try again."
        fi
    done
else

    # Otherwise we try to grab from environment variables or default to "all"

    # Corpus
    corpus=${corpus:-all}

    # Language pair
    lang_pair=${lang_pair:-all}
fi

# Don't evaluate, just predict
predict_only=${predict_only:-no}
if [ "$predict_only" == "yes" ]; then
    predict_only_flag="--predict_only"
else
    predict_only_flag=""
fi

# Backend type (vllm, transformers or other similar)
backend_type=${backend_type:-vllm}

# Output folder
results_folder="${results_folder:-./results/saminmt-llm}"
model_slug=${model_uri//\//__}
model_uid=$(cut -f1 -d/ - <<<"${model_uri}")
model_name=$(cut -f2 -d/ - <<<"${model_uri}")
slug_results_folder=${results_folder}/${model_slug}
slug_results_folder_onlymodel=${results_folder}/${model_name}

# Log file
temp_log_file=$(mktemp --suffix .log)
log_file=${slug_results_folder}/eval-$(date +%s).log

# Task name suffix
# If model URI contains "madlad" then we need to append _madlad to the task name
if [[ ${model_uri} == *"madlad"* ]]; then
    task_name_suffix=madlad
    backend_type="hf"
else
    task_name_suffix=default
fi
task_name="saminmt_${corpus}_${lang_pair}_${task_name_suffix}"
task_name=${task_name//all_all/all}

# Max batch size
max_batch_size=${max_batch_size:-4}

print_settings () {
    echo
    echo "Sami NMT evaluation:"
    echo "===================="
    echo
    echo "* Corpus: ${corpus}"
    echo "* Language pair: ${lang_pair}"
    echo "* Task name: ${task_name}"
    echo
    echo "* Model URI: ${model_uri}"
    echo "  - User ID: ${model_uid}"
    echo "  - Model name: ${model_name}"
    echo
    echo "* Backend type: ${backend_type}"
    echo "* Max batch size: ${max_batch_size}"
    echo
    echo "* Predict only: ${predict_only}"
    echo
    echo "* Output folder: ${slug_results_folder_onlymodel}"
    echo
}

# If interactive, prompt user to confirm whether to run
if [ "$interactive_mode" == "yes" ]; then
    print_settings
    read -p "Run evaluation? (y/n): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Evaluation cancelled by user."
        exit 0
    fi
fi

ngpus () {
    echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}'
}

# Main evaluation command
(

    # Log files should include the same settings info
    print_settings

    # If there are more than 1 GPU and if the backend is hf we can
    # use accelerate to leverage data or model parallelism
    if [ $(ngpus) -gt 1 ] && [ "${backend_type}" == "hf" ] && [ "${split_model_multiple_gpus}" == "yes" ]; then
        echo "Using accelerate to split model across $(ngpus) GPUs" | tee -a ${temp_log_file}
        lm_eval \
            --model ${backend_type} \
            --model_args pretrained=${model_uri},dtype=bfloat16,parallelize=True \
            --tasks ${task_name} \
            --batch_size auto \
            --max_batch_size ${max_batch_size} \
            --output_path ${slug_results_folder} \
            ${predict_only_flag} \
            --log_samples
    elif [ $(ngpus) -gt 1 ] && [ "${backend_type}" == "hf" ]; then
        echo "Using accelerate for data parallelism across $(ngpus) GPUs" | tee -a ${temp_log_file}
        accelerate launch -m lm_eval \
            --model ${backend_type} \
            --model_args pretrained=${model_uri},dtype=bfloat16 \
            --tasks ${task_name} \
            --batch_size ${max_batch_size} \
            --output_path ${slug_results_folder} \
            ${predict_only_flag} \
            --log_samples
    elif [ $(ngpus) -gt 1 ] && [ "${backend_type}" == "vllm" ]; then
        lm_eval \
            --model ${backend_type} \
            --model_args pretrained=${model_uri},tensor_parallel_size=$(ngpus),dtype=auto,gpu_memory_utilization=0.8 \
            --tasks ${task_name} \
            --batch_size auto \
            --max_batch_size ${max_batch_size} \
            --output_path ${slug_results_folder} \
            ${predict_only_flag} \
            --log_samples
    else
        lm_eval \
            --model ${backend_type} \
            --model_args pretrained=${model_uri},dtype=bfloat16 \
            --tasks ${task_name} \
            --batch_size auto \
            --max_batch_size ${max_batch_size} \
            --output_path ${slug_results_folder} \
            ${predict_only_flag} \
            --log_samples
    fi

) 2>&1 | tee -a ${temp_log_file}

# Move the temp log file to the final log file location
mv ${temp_log_file} ${log_file}
rm ${temp_log_file}

# Task groups
# saminmt_all_default
# saminmt_all_madlad
# saminmt_uit_all_default
# saminmt_uit_all_madlad
# saminmt_yle_all_default
# saminmt_yle_all_madlad

# Individual tasks
# saminmt_uit_fin-sme_default
# saminmt_uit_fin-sme_madlad
# saminmt_uit_sme-fin_default
# saminmt_uit_sme-fin_madlad
# saminmt_yle_fin-sme_default
# saminmt_yle_fin-sme_madlad
# saminmt_yle_sme-fin_default
# saminmt_yle_sme-fin_madlad
