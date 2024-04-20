evaluated_model="...NAME OF MODEL..." 

mkdir playground
cd evaluation/extraction

python extract_attribute_object.py -ip "../../generated_captions/attribute_object/${evaluated_model}_caption.json" \
        -op "../../playground/extracted_attribute_object_${evaluated_model}_gpt4.jsonl"

cd ../matching

python match_attribute_object.py -ip "../../playground/extracted_attribute_object_${evaluated_model}_gpt4.jsonl" \
        -op "../../playground/matched_attribute_object_${evaluated_model}_gpt4.jsonl"

cd ../postprocess

python postprocess_attribute_object.py -c "../../playground/matched_attribute_object_${evaluated_model}_gpt4.jsonl" \
        -o "../../playground/matched_attribute_object_${evaluated_model}_gpt4_postprocessed.jsonl"

cd ../scorer

python valorvalor_eval_attribute_object.py -i "../../playground/matched_attribute_object_${evaluated_model}_gpt4_postprocessed.jsonl" \
 -o "../../playground/valor_results_attribute_object/"

 