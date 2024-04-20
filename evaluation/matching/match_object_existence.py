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

        Task: Match objects from two lists based on their meaning.

        Input Lists:
        1. "gt_objects": A list of objects in the image.
        2. "generated_objects": A list of objects identified by a vision language model.

        For each object in "generated_objects", find the object in the "gt_objects" that have the same meaning and add it to the "matched_objects" dictionary.
        By the same meaning, we mean the words can be synonyms, can be plural/singular forms of each other and can also have different length of words to express the same meaning of objects, etc.
        Note since we find the matched object for each object in "generated_objects", it's ok that multiple objects in "generated_objects" match one object in "gt_objects", list all matches.
        There is special scenario that when you can't find the matched object in "gt_objects" but you can find one or more object is a subset or a sub category of the generated object, which means that the generated object is a broader concept of the object in "gt_objects", add it to the "broader_concept" dictionary instead of the "matched_objects". If there are many objects are a subset or a sub category of the generated object, you can pick anyone of them. Note we are matching for each object in "generated_objects". If you can find the matched object in "gt_objects", you should not add it to the "broader_concept" dictionary.
        Note that broader concept doesn't mean more of the same object, it means a broader category of the object. For example, "plant" is a broader concept of "flower" and "food" is a broader concept of "broccoli". However, since 'people' is plural of 'person', 'people' is not a broader concept of 'person'.
        
        Output:
        - A "broader_concept" dictionary: only if an object from "generated_objects" denotes a broader category of a concept in "gt_objects", not vice versa, like "plant" is a broader conecpt of "flower" and "food" is a broader concept of "broccoli". Key = word from "generated_objects", Value = word from "gt_objects".
        - A "matched_objects" dictionary: only if an object from "generated_objects can be mapped to an object in "gt_objects" with the matching criteria. Key = word from "generated_objects", Value = word from "gt_objects". It should not contain any words from the "broader_concept" dictionary.

        For clarity, consider these examples:
        ### Example 1:
        - gt_objects: ["bench", "fence", "bell", "clock", "doorway", "hat", "shoe", "steps", "building", "roof", "people", "woman", "pigeon", "tree", "sky", "clouds", "dress"]
        - generated_objects: ["woman", "bench", "church", "hat", "people", "man", "child", "fountain", "water", "clock"]
        - broader_concept: {}
        - matched_objects:{"woman": "woman", "bench": "bench", "hat": "hat", "people": "people", "clock": "clock"}
        ### Example 2:
        - gt_objects: ["sock", "person", "trees", "players", "outfit", "ball", "guys", "uniform", "man", "guy", "field", "frisbee","street","sign","people", "shoe", "tree", "player", "arm"]
        - generated_objects: ["men", "field", "player", "people", "spectators", "supporters", "person", "road", "street signs"]
        - broader_concept: {}
        - matched_objects: {"men": "man, "field": "field", "player": "player", "people": "people", "person": "person", 'road': 'street' 'street signs': 'sign'}
        ### Example 3:
        - gt_words: ["vase", "flowers", "wall", "staircase", "doors", "lamps", "pillows", "bedroom", "comforter", "lamp", "bed", "flowers", "sheets", "headboard", "wall", "ceiling", "nightstand"] 
        - generated_words: ["bedroom", "bed", "pillows", "lamps", "plants", "potted plants", "vase", "nightstand"] 
        - broader_concept: {"plants": "flowers", "potted plants": "flowers"} 
        - matched_objects: {"bedroom": "bedroom", "bed": "bed", "pillows": "pillows", "lamps": "lamps", "vase": "vase", "nightstand": "nightstand"}
        ### Example 4:
        - gt_words: ["hand", "game controller", "pants", "man", "arm", "earring", "woman", "can", "table", "hair", "woman", "shirt", "coffee table", "glasses", "water", "candle", "pole", "t-shirt", "shorts", "sofa"] 
        - generated_words: ["people", "couch", "men", "woman", "Wii remote", "screen", "cups", "coffee table", "bottle", "potted plant"]
        - broader_concept: {"people": "man", "people": "woman"}
        - matched_objects: {"men": "man", "coffee table": "coffee table", "woman": "woman", "Wii remote": "game controller"}
        ### Example 5:
        - gt_words: ['curtains', 'toy', 'wall', 'pajamas', 'water bottle', 'boy', 'lid', 'bed', 'headboard', 'woman', 'mother', 'shirt', 'desk']
        - generated_words: ['child', 'bed', 'pajamas', 't-shirt', 'toys', 'teddy bear', 'doll', 'stuffed animal', 'chair', 'table']
        - broader_concept: {'child': 'boy'}
        - matched_objects: {'mother': 'mother', 'bed': 'bed', 'pajamas': 'pajamas', 't-shirt': 'shirt', 'table': 'desk'}
        ### Example 6:
        - gt_words: ['ground', 'train tracks', 'smoke', 'bridge', 'moss', 'train', 'steam', 'coat', 'tree']
        - generated_words: ['train', 'stone arch bridge', 'dirt road', 'people', 'jacket', 'person', 'cars', 'tree', 'trees']
        - broader_concept: {}
        - matched_objects: {'train': 'train', 'stone arch bridge': 'bridge', 'jacket':'coat', 'tree': 'tree', 'trees': 'tree'}}
        ### Example 7:
        - gt_words: ['blender','countertop', 'microwave', 'cabinets', 'dishwasher', 'sink', 'toaster', 'counter', 'water bottle', 'window', 'sign', 'door', 'bar stool', 'bar stool', 'kitchen', 'refrigerator']
        - generated_words: ['kitchen', 'cabinets', 'countertops', 'refrigerator', 'microwave oven', 'dishwasher', 'bottles', 'counter', 'wine glass', 'chairs']
        - broader_concept: {'chairs': 'bar stool', 'bottles': 'water bottle'}
        - matched_objects: {'kitchen': 'kitchen', 'cabinets': 'cabinets', 'counter': 'counter', 'refrigerator': 'refrigerator', 'microwave oven': 'microwave', 'dishwasher': 'dishwasher', 'countertops': 'countertop'}
       
        With these examples in mind, please help me extract the broader_concept, and matched_objects from the following two objects lists.
        ```
        - gt_objects: [GT_OBJECTS]
        - generated_objects: [GENERATED_OBJECTS]
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
        
        generated_objects = list(generated_objects)

        content = prompt_template.replace("[GT_OBJECTS]", str(gt_objects))
        content = content.replace("[GENERATED_OBJECTS]", str(generated_objects))    

        prompt = [{"role": "user", "content": content}]
        llm_output_dict = llm(prompt)

        output_file[image_id] = llm_output_dict
        current_output = {
            "image_id": image_id,
            "gt_words": gt_objects,
            "generated_words": generated_objects,
            "broader_concept": llm_output_dict["broader_concept"],
            "matched_objects": llm_output_dict["matched_objects"],
            "caption": cap["generated_caption"],
        }
        print(current_output)
        with open(args.output_file_path, "a") as f:
            f.write(json.dumps(current_output) + "\n")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="objects matching")
    parser.add_argument("-ip", "--input_file_path", type=str, required=True, help="Path to the input file") #expect jsonl format
    parser.add_argument("-op", "--output_file_path", type=str,required=True, help="Path to save the output file") #expect jsonl format
    parser.add_argument("-r", "--random", default=False, action='store_true')
    parser.add_argument("-n", "--num_random_samples", type=int, default=5)
    args = parser.parse_args()
    main(args)