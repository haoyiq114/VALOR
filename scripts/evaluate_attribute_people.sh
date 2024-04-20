evaluated_model="...NAME OF MODEL..." 

mkdir playground
cd evaluation/extraction

python extract_attribute_people.py -ip "../../generated_captions/attribute_people/${evaluated_model}_caption.json" \
        -op "../../playground/extracted_attribute_people_${evaluated_model}_gpt4.jsonl"

cd ../matching

python match_attribute_people.py -ip "../../playground/extracted_attribute_people_${evaluated_model}_gpt4.jsonl" \
        -op "../../playground/matched_attribute_people_${evaluated_model}_gpt4.jsonl"

cd ../postprocess

python postprocess_attribute_people.py -c "../../playground/matched_attribute_people_${evaluated_model}_gpt4.jsonl" \
        -o "../../playground/matched_attribute_people_${evaluated_model}_gpt4_postprocessed.jsonl"

cd ../scorer

python valorvalor_eval_attribute_people.py -i "../../playground/matched_attribute_people_${evaluated_model}_gpt4_postprocessed.jsonl" \
 -o "../../playground/valor_results_attribute_people/"

 