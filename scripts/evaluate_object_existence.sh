evaluated_model="...NAME OF MODEL..." 

mkdir playground
cd evaluation/extraction

python extract_object_existence.py -ip "../../generated_captions/object_existence/${evaluated_model}_long_caps.json" \
        -op "../../playground/extracted_object_existence_${evaluated_model}_gpt4.jsonl"

cd ../postprocess

python postprocess_object_existence.py -c "../../playground/extracted_object_existence_${evaluated_model}_gpt4.jsonl" \
        -o "../../playground/extracted_object_existence_${evaluated_model}_gpt4_postprocessed.jsonl"

python prepare_before_match_object_existence.py -i "../../playground/extracted_object_existence_${evaluated_model}_gpt4_postprocessed.jsonl" \
        -o "../../playground/prematch_object_existence_${evaluated_model}_gpt4.jsonl"

cd ../matching

python match_object_existence_existence.py -ip "../../playground/prematch_object_existence_${evaluated_model}_gpt4.jsonl" \
        -op "../../playground/matched_object_existence_${evaluated_model}_gpt4.jsonl"

cd ../scorer

python chair_score_object_existence.py -i "../../playground/matched_object_existence_${evaluated_model}_gpt4.jsonl" \
 -o "../../playground/chair_results_object/"

 