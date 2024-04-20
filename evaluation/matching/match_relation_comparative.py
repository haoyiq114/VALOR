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

        Task: Match the correct objects with the same meaning from a ground truth dictionary and a generated dictionary.

        Inputs:
        1. "gt_objects": A dictionary of ground truth objects. Each key is a number starting rank No. 1 and increment each time by 1. Each value is the correpsonding object with the rank.
        Sometimes one object can be, for example "ground / court, it means either ground or court is correct and should be matched. 
        2. "generated_objects": A dictionary with rank being the key and the object being the value. The rank is the rank of the object in the generated caption.

        Instructions:
        1. Matching Criteria:
        - For each object in "generated_objects", find the object in the "gt_objects" that have the same meaning and add it to the "matched_objects" dictionary.
        - By the same meaning, we mean the words can be synonyms, can be plural/singular forms of each other and can also have different length of words to express the same meaning of objects, etc. 
        - Notice that the final macthed_objects must follow the order of values in "generated_objects". 
        - If you find that the "generated_objects" can be matched with the "gt_objects" but the object in "generated_objects" is a broader concept of the objects in "gt_objects", for example, one object in "generated_objects" is "person", but the "gt_objects" don't have "person" but specifically have "man", which is a subcategory of "person", add it to the "broader_concept" dictionary instead of the "matched_objects".    
        2. Output:
        - A "broader_concept" dictionary: only if an object from "generated_objects" denotes a broader category of a concept in "gt_objects" Notify that Key must be the item from "generated_objects", Value must be item from "gt_objects". If none, it should be an empty dictionary.
        - A "matched_objects" dictionary: only if an object from "generated_objects" can be mapped to an object in "gt_objects" with the matching criteria. Key must be word from "generated_objects", Value must be word from "gt_objects". It should not contain any words from the "broader_concept" dictionary.

        For clarity, consider these examples:
        
        ### Example 1:
        - gt_objects:{"1": "stove", "2": "table", "3": "cabinet", "4": "bottle / wine", "5": "mug / cup / coffee", "6": "glass", "7": "spoon"}
        - generated_objects: {'1': 'bottle', '2': 'mug', '3': 'spoon', '4': 'stove'}
        - broader_concept: {}
        - matched_objects: {'bottle': 'bottle / wine', 'mug': 'mug / cup / coffee', 'spoon': 'spoon', 'stove': 'stove'}
        
        ### Example 2:
        - gt_objects: {"1": "grass ground", "2": "tennis court", "3": "net background", "4": "man", "5": "woman", "6": "chair", "7": "racket"}
        - generated_objects: {"1": "man", "2": "tennis racket", "3": "chairs", "4": "person", "5": "ball"}
        - broader_concept: {"person": "man"}
        - matched_objects: {"man": "man", "tennis racket": "racket", "chairs": "chair"}
        Notice the broader_concept here can either be {"person": "man"} or {"person": "woman"} as both man and woman is a sub category of person. 
        
        With these examples in mind, please help me extract the broader_concept, and matched_objects from the following two inputs.
        ```
        - gt_objects: [GT_RELATIONS]
        - generated_objects: [GENERATED_RELATIONS]
        ```
        Please note that your answer should only be in a JSON format with the keys only being the broader_concept, and matched_objects.
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
        
        if generated_objects == "":
            current_output = {
                "image_id": image_id,
                "gt_words": gt_objects,
                "generated_words": generated_objects,
                "broader_concept": {},
                "matched_objects": {},
                "generated_caption": generated_caption,
            }
        else: 
            llm_output_dict = llm(prompt)

            output_file[image_id] = llm_output_dict
            current_output = {
                "image_id": image_id,
                "gt_words": gt_objects,
                "generated_words": generated_objects,
                "broader_concept": llm_output_dict["broader_concept"],
                "matched_objects": llm_output_dict["matched_objects"],
                "generated_caption": generated_caption,
            }
            
        print("broader_concept", llm_output_dict["broader_concept"])
        print("matched_objects", llm_output_dict["matched_objects"])
        with open(args.output_file_path, "a") as f:
            f.write(json.dumps(current_output) + "\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="objects matching")
    parser.add_argument("-ip", "--input_file_path", type=str,required=True)
    parser.add_argument("-op", "--output_file_path", type=str, required=True)
    parser.add_argument("-r", "--random", type=bool, default=False)
    parser.add_argument("-n", "--num_random_samples", type=int, default=50)
    args = parser.parse_args()
    main(args)