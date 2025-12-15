#!/usr/bin/env bash

grab_data () {
    jq -c '
        . | {
            "sentence_id": .doc.sentence_id,
            "text_fin": .doc.text_fin,
            "text_sme": .doc.text_sme,
            "hyp": .filtered_resps[0]
        }
    '
}

json2yaml () {
    yq -p json -o yaml
}

pretty_print () {
    bat --language yaml
}

input_file=${1:-/dev/stdin}

cat ${input_file} | grab_data | json2yaml | pretty_print
