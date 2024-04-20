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

    Task: Match the correct (attribute, object) with the same meaning from a ground truth dictionary and a generated dictionary.

    Inputs:
    1. "gt_att_obj": A dictionary of ground truth (attribute, object) pairs. Each key is a number starting from 1 and increment each time by 1. Each value is the corresponding (attribute, object) paird related to a person. 
    Sometimes one object can be, for example "(black, bag), (white, bag), (striped, bag)", it means either "black" or "white" or "striped" is correct for an attribute related with the "bag" and should be matched. 
    2. "generated_att_obj": A dictionary with order being the key and the (attribute, object) pairs related to a person being the value. The order is the order of the object in the generated caption.

    Instructions:
    1. Matching Criteria:
    - For each (attribute, object) in "generated_att_obj", find the (attribute, object) in the "gt_att_obj" that have the same meaning and add it to the "matched_att_obj" dictionary. Remeber these matched (attribute, object) pairs should be related to the same person.
    - By the same meaning, we mean the words can be synonyms, can be plural/singular forms of each other and can also have different length of words to express the same meaning of attributes or objects, etc. 
    - Notice that the final "matched_att_obj" must follow the order of values in "generated_att_obj". 
    - If you find that the "generated_att_obj" can be matched with the "gt_att_obj" but the attribute or object in "generated_att_obj" is a broader concept of the attribute or object in "gt_att_obj", for example, one object in "generated_att_obj" is "person", but the "gt_att_obj" don't have "person" but specifically have "man", which is a subcategory of "person", add it to the "broader_concept" dictionary instead of the "matched_att_obj".
    2. Output:
    - A "broader_concept" dictionary: {"ORDER": {"person": {"PERSON1": "PERSON2"}, "object": {"(ATTRIBUTE1, OBJECT1)": "(ATTRIBUTE2, OBJECT2)", "(ATTRIBUTE1, OBJECT1)": "(ATTRIBUTE2, OBJECT2)", ...}} only if a PERSON1 from "generated_att_obj" matched a PERONS2 from "gt_att_obj" and (ATTRIBUTE1, OBJECT1) from "generated_att_obj" denotes a broader category of an (ATTRIBUTE2, OBJECT2) in "gt_att_obj". Notify that Keys must be the (ATTRIBUTE1, OBJECT1) from "generated_att_obj", Values must be (ATTRIBUTE2, OBJECT2) from "gt_att_obj". If none, it should be an empty dictionary.
    - A "matched_att_obj" dictionary: {"ORDER": {"person": {"PERSON1": "PERSON2"}, "object": {"(ATTRIBUTE1, OBJECT1)": "(ATTRIBUTE2, OBJECT2)", "(ATTRIBUTE1, OBJECT1)": "(ATTRIBUTE2, OBJECT2)", ...}} only if a PERSON1 from "generated_att_obj" matched a PERONS2 from "gt_att_obj" and an (ATTRIBUTE1, OBJECT1) from "generated_att_obj" can be mapped to an (ATTRIBUTE2, OBJECT2) in "gt_att_obj" with the matching criteria. Keys must be (ATTRIBUTE1, OBJECT1) from "generated_att_obj", Values must be (ATTRIBUTE2, OBJECT2) from "gt_att_obj". It should not contain any (ATTRIBUTE1, OBJECT1) or (ATTRIBUTE2, OBJECT2) from the "broader_concept" dictionary.
    - One key can contain a list of matched (ATTRIBUTE, OBJECT) pairs.
    - In "broader_concept" and "matched_att_obj", you cannot map two different (ATTRIBUTE1, OBJECT1) from "generated_att_obj" to the same (ATTRIBUTE2, OBJECT2) from "gt_att_obj" twice.
    - REMEMBER both "broader_concept" and "matched_objects" should contain the person information.

    For clarity, consider these examples:
            
    ### Example 1:
    - gt_att_obj: {"1": {"person": "woman", "object": "(black, blazer), (pink, pants), (brown, shoes), (green, hat)", "action": "standing, hugging"}, "2": {"person": "man", "object": "(red, scarf), (red, jacket), (white, pants), (brown, purse), (black, shoes)", "action": "standing"}}
    - generated_att_obj: {"1": {"person": "man", "object": "(red, jacket), (brown, handbag), (red, scarf)", "action": "holding"}, "2": {"person": "woman", "object": "(black, jacket), (pinkish, pants), (brown, purse), (green, hat)", "action": "being held"}}
    - broader_concept: {"1": {"person": {"man": "man"}, "object": {"(brown, handbag)": "(brown, purse)"}, "2": {"person": {"woman": "woman"}, "object": {"(black, jacket)": "(black, blazer)"}}}
    - matched_objects: {"1": {"person": {"man": "man"}, "object": {"(red, jacket)": "(red, jacket)", "(red, scarf)": "(red, scarf)"}}, "2": {"person": {"woman": "woman"}, "object": {"(pinkish, pants)": "(pink, pants)", "(green, hat)": "(green, hat)"}}}

    With these examples in mind, please help me extract the broader_concept, and matched_objects from the following two inputs. Please note that your answer should only be in a JSON format with the keys only being the broader_concept, and matched_att_obj.  No explanations/note included.
    ```
    - gt_att_obj: [GT_ATT_OBJ]
    - generated_att_obj: [GENERATED_ATT_OBJ]
    ```  

    """

    with open(args.gt_file_path, "r") as f:
        gt_cap_info = json.load(f)
    
    with open(args.generated_file_path, "r") as f:
        generated_cap_info = [json.loads(l) for l in f.readlines()]

    
    # if args.random:
    #     cap_info = random.sample(cap_info, args.num_random_samples)

    output_file = {}

    messages = [
        {
            "role": "system", 
            "content": "You are a language assistant that helps to extract information from given sentences." 
        },
    ]
    
    for cap in tqdm.tqdm(generated_cap_info):
        image_id = cap["image_id"]
        generated_obj_atts = cap["response"]["clothes"]
        gt_obj_atts = gt_cap_info[image_id]
        
        content = prompt_template.replace("[GT_ATT_OBJ]", str(gt_obj_atts))
        content = content.replace("[GENERATED_ATT_OBJ]", str(generated_obj_atts))    

        prompt = messages + [{"role": "user", "content": content}]
        llm_output_dict = llm(prompt, args.gpt_model)
    
        print(image_id, llm_output_dict)
        output_file[image_id] = llm_output_dict
        current_output = {
            "image_id": image_id,
            "gt_obj_atts": gt_obj_atts,
            "generated_obj_atts": generated_obj_atts,
            "matched_att_obj": llm_output_dict["matched_att_obj"],
            "broader_concept": llm_output_dict["broader_concept"]
        }
        
        with open(args.output_file_path, "a") as f:
            f.write(json.dumps(current_output) + "\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="objects matching")                                          
    parser.add_argument("-igtp", "--gt_file_path", type=str, required=True)
    parser.add_argument("-igep", "--generated_file_path", type=str, required=True)
    parser.add_argument("-op", "--output_file_path", type=str, required=True)
    parser.add_argument("-m", "--gpt_model", type=str, default="gpt-3.5-turbo")
    # parser.add_argument("-r", "--random", type=bool, default=False)
    # parser.add_argument("-n", "--num_random_samples", type=int, default=50)

    args = parser.parse_args()
    main(args)
