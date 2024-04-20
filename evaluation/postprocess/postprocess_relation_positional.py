import json
import argparse
from copy import deepcopy

def postprocess(image_id, objects):
    
    # Unwanted relations
    fake_ralations = ["complements", "complement", "towards", "through", #'wearing' 
                      "holding", "using", "appears", "penetrating", "reads", "separated", "has"
                    ]
    
    filtered_objects = []
    for obj in objects:
        #filter action word ending with "ing", might use nltk to improve this? 
        # if obj.endswith("ing"):
        #     objects.remove(obj)
        #     next
        words = obj.split()
        for word in words:
            if word in fake_ralations:
                filtered_objects.append(obj)
    
    print(image_id, f"Filtered {len(filtered_objects)} objects", filtered_objects)             
    returned_objects = [obj for obj in objects if obj not in filtered_objects]

    return returned_objects


def main(args):    
    with open(args.caption_path, "r") as f:
        captions = [json.loads(l) for l in f.readlines()]
    out = deepcopy(captions)
        
    for idx, item in enumerate(captions):
        image_id = item["image_id"]
        objects = item["response"]["relations"]
        objects = postprocess(image_id, objects)
        out[idx]["response"]["relations"] = objects
    
    with open(args.output_path, "a") as outfile:
        for entry in out:
            json.dump(entry, outfile)
            outfile.write('\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess the generated captions")
    parser.add_argument("-c", "--caption_path", type=str, required=True, help="Path to the generated captions") #expect jsonl format
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save the postprocessed captions") #expect jsonl format
    
    args = parser.parse_args()
    main(args)