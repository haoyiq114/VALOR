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
        1. "gt_att_obj":A dictionary with order being the key and the ground-truth (attribute, object) pair being the value. The order is the order of the object in the image.
        Sometimes one object can be, for example "(black, bag), (white, bag), (striped, bag)", it means either "black" or "white" or "striped" is correct for an attribute related with the "bag" and should be matched. 
        2. "generated_att_obj": A dictionary with order being the key and the generated (attribute, object) pair being the value. The order is the order of the object in the generated caption.

        Instructions:
        1. Matching Criteria:
        - For each (attribute, object) in "generated_att_obj", find the (attribute, object) in the "gt_att_obj" that have the same meaning and add it to the "matched_att_obj" dictionary.
        - By the same meaning, we mean the words can be synonyms, can be plural/singular forms of each other and can also have different length of words to express the same meaning of attributes or objects, etc. 
        - If you find that the "generated_att_obj" can be matched with the "gt_att_obj" but the attribute or object in "generated_att_obj" is a broader concept of the attribute or object in "gt_att_obj", for example, one object in "generated_att_obj" is "person", but the "gt_att_obj" don't have "person" but specifically have "man", which is a subcategory of "person", add it to the "broader_concept" dictionary instead of the "matched_att_obj".
        2. Output:
        - A "broader_concept" dictionary: {"ORDER2": {"(ATTRIBUTE1, OBJECT1)": "(ATTRIBUTE2, OBJECT2)"}} only if an (ATTRIBUTE1, OBJECT1) with ORDER1 from "generated_att_obj" denotes a broader category of an (ATTRIBUTE2, OBJECT2) with ORDER2 in "gt_att_obj". Notify that Key must be the (ATTRIBUTE1, OBJECT1)from "generated_att_obj", Value must be (ATTRIBUTE2, OBJECT2) from "gt_att_obj". If none, it should be an empty dictionary. ORDER1 should be the same as ORDER2.
        - A "matched_att_obj" dictionary: {"ORDER2": {"(ATTRIBUTE1, OBJECT1)": "(ATTRIBUTE2, OBJECT2)"}} only if an (ATTRIBUTE1, OBJECT1) with ORDER1 from "generated_att_obj" can be mapped to an (ATTRIBUTE2, OBJECT2) with ORDER2 in "gt_att_obj" with the matching criteria. Key must be (ATTRIBUTE1, OBJECT1) from "generated_att_obj", Value must be (ATTRIBUTE2, OBJECT2) from "gt_att_obj". It should not contain any (ATTRIBUTE1, OBJECT1) or (ATTRIBUTE2, OBJECT2) from the "broader_concept" dictionary. ORDER1 should be the same as ORDER2.
        - The keys in "broader_concept" and "matched_att_obj" must be the same as "gt_att_obj".
        - In "broader_concept" and "matched_att_obj", you can not map two different (ATTRIBUTE1, OBJECT1) from "generated_att_obj" to the same (ATTRIBUTE2, OBJECT2) from "gt_att_obj" twice.

        For clarity, consider these examples:
        
        ### BAD Example 1:
        - gt_att_obj: {"1": "(green, lollipop), (yellow, lollipop), (white, stick)", "2": "(orange, candy)", "3": "(green, candy), (yellow, candy), (white, stick)", "4": "(orange, lollipop)", "5": "(blue, lollipop)", "6": "(pink, background)"}
        - generated_att_obj: {"1": "(green, candy)", "2": "(orange, lollipop)", "3": "(yellow, lollipop)", "4": "(light green, candy)", "5": "(pastel pink, background)"}}
        - broader_concept: {"1": {"(green, candy)": "(green, lollipop)"}, "4": {"(light green, candy)": "(green, lollipop)"} }
        - matched_objects: {"2": {"(orange, lollipop)": "(orange, candy)"}, "3": {"(yellow, candy)": "(yellow, lollipop)", "5": {"(pastel pink, background)": "(pink, background)"}}}}
        This is a BAD example because in "broader_concept", you can not map both (green, candy) and (light green, candy) to (green, lollipop) since there is only one (green, lollipop) in "gt_att_obj". In this case, you don't need to map these green-ish objects in "generated_att_obj".
        You cannot map (pastel pink, background) to (pink, background) because they do not have the same order.
  
        A good example is 
        - broader_concept: {"1": {"(green, candy)": "(green, lollipop)"}}
        - matched_objects: {"2": {"(orange, lollipop)": "(orange, candy)"}, "3": {"(yellow, candy)": "(yellow, lollipop)"}}}

        ### BAD Example 2:
        - gt_att_obj: {"1": "(green, candy)", "2": "(white, candy)", "3": "(yellow, candy)", "4": "(green, candy)"}
        - generated_att_obj: {"1": "(green, lollipop)", "2": "(white, lollipop)", "3": "(yellow, lollipop)", "4": "(yellow, lollipop)", "5": "(light yellow, lollipop)", "6": "(green, lollipop)", "7": "(dark yellow, lollipop)"}
        - broader_concept: {}
        - matched_objects: {"1": {"(green, lollipop)": "(green, candy)"}, "2": {"(white, lollipop)": "(white, candy)"}, "3": {"(yellow, lollipop)": "(yellow, candy)"}, "4": {"(light yellow, lollipop)": "(yellow, candy)"}, "5": {"(green, lollipop)": "(green, candy)"}}
        This is a BAD example because in "matched_object", you cannot map both (yellow, lollipop) and (light yellow, lollipop) to (yellow, candy) since there is only one (yellow, candy) in "gt_att_obj". In this case, you don't need to map these yellow-ish objects in "generated_att_obj".
        
        A good example is:
        - matched_objects: {"1": {"(green, lollipop)": "(green, candy)"}, "2": {"(white, lollipop)": "(white, candy)"}, "3": {"(yellow, lollipop)": "(yellow, candy)"}, "4": {"(green, lollipop)": "(green, candy)"}}

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

    output_file = {}

    messages = [
        {
            "role": "system", 
            "content": "You are a language assistant that helps to extract information from given sentences." 
        },
    ]
    
    for cap in tqdm.tqdm(generated_cap_info):
        image_id = cap["image_id"]
        generated_obj_atts = cap["response"]["objects"]
        gt_obj_atts = gt_cap_info[image_id]
        
        content = prompt_template.replace("[GT_ATT_OBJ]", str(gt_obj_atts))
        content = content.replace("[GENERATED_ATT_OBJ]", str(generated_obj_atts))    

        prompt = messages + [{"role": "user", "content": content}]
        llm_output_dict = llm(prompt, args.gpt_model)
    
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

    args = parser.parse_args()
    main(args)