#!/usr/bin/env bash

[ $# -lt 1 ] && echo "Please include JSON file as 1st argument" && exit

# NOTE: expects JSONL file to be named `sentences-<source>-<target>.jsonl`

export jsonl_file=$1
export jsonl_output_file=${jsonl_file//.jsonl/-pretty.jsonl}
export lang_pair=$(basename $jsonl_file | cut -f2,3 -d- | sed 's/.jsonl$//')
export src_lang=$(echo $lang_pair | cut -f1 -d-)

declare -A flores_codes=(
  [de]="deu_Latn"  # German
  [en]="eng_Latn"  # English
  [es]="spa_Latn"  # Spanish
  [fr]="fra_Latn"  # French
  [it]="ita_Latn"  # Italian
  [ko]="kor_Hang"  # Korean
  [nl]="nld_Latn"  # Dutch
  [pt]="por_Latn"  # Portuguese
  [ru]="rus_Cyrl"  # Russian
)

export lang_code=${flores_codes[$src_lang]}

cat ${jsonl_file} | jq -cr """
{
    \"ref\": .target, 
    \"hyp\": .filtered_resps[0],
    \"src\": .doc.sentence_${lang_code}
}
""" \
| jq -cr '{"ref": .ref, "src": .src, "hyp": .hyp | sub("^\\s"; "")}' \
> ${jsonl_output_file}
