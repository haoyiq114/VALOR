evaluated_model="...NAME OF MODEL..." 

mkdir playground
cd evaluation/extraction

python extract_relation_comparative.py -ip "../../generated_captions/relation_comparative/${evaluated_model}_comp.json" \
        -op "../../playground/extracted_objects_${evaluated_model}_gpt4.jsonl"

cd ../postprocess

python prepare_before_match_relation.py -i "../../playground/extracted_objects_${evaluated_model}_gpt4.jsonl" \
        -o "../../playground/prematch_objects_${evaluated_model}_gpt4.jsonl" \
        -gt "../../human_annotation/relation_comparative.json"

cd ../matching

python match_relation_comparative.py -ip "../../playground/prematch_objects_${evaluated_model}_gpt4.jsonl" \
        -op "../../playground/matched_objects_${evaluated_model}_gpt4.jsonl"

cd ../postprocess

python postprocess_relation_comparative.py -ip "../../playground/matched_objects_${evaluated_model}_gpt4.jsonl" \
        -op "../../playground/matched_objects_${evaluated_model}_gpt4_postprocessed.jsonl"

cd ../scorer

python chair_score_relation_comparative.py -i "../../playground/matched_objects_${evaluated_model}_gpt4_postprocessed.jsonl" \
 -o "../../playground/chair_results_object/"

 

 