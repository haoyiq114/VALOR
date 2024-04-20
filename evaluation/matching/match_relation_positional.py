import os
import json
import tqdm
import random
import openai
import argparse
import sys
sys.path.append("../")
from gpt_model import llm, set_key

random.seed(1234)

def main(args):
    set_key()
        
    prompt_template = """

        Task: Match (object-1 positional relation with object-2) from a ground truth dictionary and a list based on their meaning.

        Inputs:
        1. "gt_relations": A dictionary of ground truth relations. Each key is a number with no meaning of order. Each key represents different relations. The values is a list of one or two relations, 
        if there are two relations, they are synonyms. Sometimes in one relation it contains for example "image / table", it means either image or table in this phrase is correct. 
        2. "generated_relations": A list of generated relations from a model.

        Instructions:
        1. Matching Criteria:
        - For each relation in "generated_relations", find the corresponding relation in "gt_relations" based on their meaning, if there is none, skip it.
        - If you find a match, add it to the "matched_relations" dictionary. Note that if there are two relations in a item of "gt_relations", 
        it means the same meaning of the relation, you can pick either one of them as the match to the relation in "generated_relations".
        - If you find that the generated relation is a broader concept of a relation in "gt_relations" such as the generated relation is near each other, next to, in touch etc. 
        but the gt_relation specifically have their relation is specifically left, right, behind or front, etc, which is more than near, add it to the "broader_concept" dictionary.
        2. Output:
        - A "broader_concept" dictionary: only if an relation from "generated_relations" denotes a broader category of a concept in "gt_relations" Notify that Key must be the item from "generated_relations", Value must be item from "gt_relations". If none, it should be an empty dictionary.
        - A "matched_relations" dictionary: only if an relation from "generated_relations can be mapped to an relation in "gt_relations" with the matching criteria. Key must be word from "generated_relations", Value must be word from "gt_relations". It should not contain any words from the "broader_concept" dictionary.

        For clarity, consider these examples:
        ### Example 1:
        - gt_relations:{"0": ["the television is on the right side of the image"], "1": ["leaves are on the left side of the image"], "2": ["the television is on the ground", "the ground is under the television"], "3": ["leaves are on the ground", "the ground is under the leaves"], "4": ["the television is to the right of the leaves", "the leaves are to the left of the television"]}
        - generated_relations: ["an old television set is on the left side of the image", "a broken television set is on the right side of the image", "a remote control is at the top of the image"]
        - broader_concept: {}
        - matched_relations:{"a broken television set is on the right side of the image": "the television is on the right side of the image"}
        
        ### Example 2:
        - gt_relations:{"0": ["the teddy bear is on the right side of the image"],"1": ["the teddy bear is on the ground"],"2": ["the teddy bear is to the right of the trash bin", "the trash bin is to the left of the teddy bear"],"3": ["the trash bin is on the ground"],"4": ["the trash bag is on the ground"],"5": ["the trash bin is to the right of the trash bag", "the trash bag is to the left of the trash bin"],"6": ["the trash bin is in the middle of the image"],"7": ["one trash bag is on the left side of the image"],"8": ["one trash bag is on the right side of the image"],"9": ["the teddy bear is to the right of the trash bag", "the trash bag is to the left of the teddy bear"]}
        - generated_relations:["a large teddy bear is on the ground", "a large teddy bear is next to a green trash can", "a green trash can is on the right side of the image", "a large teddy bear is on the left side of the image", "two black trash bags are on the ground", "one black trash bag is on the left side of the image", "one black trash bag is on the right side of the image"]
        - broader_concept: {"a large teddy bear is next to a green trash can": "the teddy bear is to the right of the trash bin"}
        - matched_relations:{"a large teddy bear is on the ground": "the teddy bear is on the ground", "two black trash bags are on the ground": "the trash bag is on the ground", "one black trash bag is on the left side of the image":"the trash bag is on the left side of the image", "one black trash bag is on the right side of the image": "one trash bag is on the right side of the image"}    
        With these examples in mind, please help me extract the broader_concept, and matched_relations from the following two inputs.
        ```
        - gt_relations: [GT_RELATIONS]
        - generated_relations: [GENERATED_RELATIONS]
        ```
        Please note that your answer should only be in a JSON format with the keys only being the broader_concept, and matched_relations.
        No explanations/note included.
    """

    with open(args.input_file_path, "r") as f:
        cap_info = [json.loads(l) for l in f.readlines()]
    
    if args.random:
        cap_info = random.sample(cap_info, args.num_random_samples)

    output_file = {}
        
    for cap in tqdm.tqdm(cap_info):
        image_id = cap["image_id"]
        gt_objects = cap["gt_words"]
        generated_objects = cap["generated_words"]
        generated_caption = cap["generated_caption"]

        content = prompt_template.replace("[GT_RELATIONS]", str(gt_objects))
        content = content.replace("[GENERATED_RELATIONS]", str(generated_objects))    

        prompt = [{"role": "user", "content": content}]
        llm_output_dict = llm(prompt)

        output_file[image_id] = llm_output_dict
        current_output = {
            "image_id": image_id,
            "gt_words": gt_objects,
            "generated_words": generated_objects,
            "broader_concept": llm_output_dict["broader_concept"],
            "matched_objects": llm_output_dict["matched_relations"],
            "generated_caption": generated_caption,
        }
        print("matched_objects", llm_output_dict["matched_relations"])
        with open(args.output_file_path, "a") as f:
            f.write(json.dumps(current_output) + "\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="objects matching")
    parser.add_argument("-ip", "--input_file_path", type=str, required=True, help="Path to the input file") #expect jsonl format
    parser.add_argument("-op", "--output_file_path", type=str,required=True, help="Path to save the output file") #expect jsonl format
    parser.add_argument("-r", "--random", type=bool, default=False)
    parser.add_argument("-n", "--num_random_samples", type=int, default=50)
    args = parser.parse_args()
    main(args)