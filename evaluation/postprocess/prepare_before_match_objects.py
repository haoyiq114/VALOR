import json
import os
import argparse

def main(args):
    gt_words = args.gt_path
    with open(gt_words, 'r') as f:
        gt_annotate = [json.loads(l) for l in f.readlines()]
    gt_words = {}
    for i, item in enumerate(gt_annotate):
        gt_words[item['image_id']] = item['gt_objects']
    
    extracted_relations = args.extracted_path
    with open(extracted_relations, 'r') as f:
        extracted_relations = [json.loads(l) for l in f.readlines()]
    outpath = args.output_path
    
    if os.path.exists(outpath):
        os.remove(outpath)
    for relation in extracted_relations:
        image_id = relation['image_id']
        if image_id in gt_words.keys():
            gt = list(set(gt_words[image_id]))
            generated = relation['objects']
            out = {
                'image_id': image_id,
                'gt_words': gt,
                'generated_words': generated,
                "generated_caption": relation['caption']
            }
            with open(outpath, "a") as f:
                f.write(json.dumps(out)+"\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare match objects")
    parser.add_argument("-gt", "--gt_path", type=str, default='../../human_annotation/object_existence.jsonl')
    parser.add_argument("-i", "--extracted_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
    