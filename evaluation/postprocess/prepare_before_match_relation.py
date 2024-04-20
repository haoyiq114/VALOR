import json
import os
import argparse

def main(args):

    gt_words = args.gt_words
    with open(gt_words, 'r') as f:
        gt_words = json.load(f)
    
    extracted_relations = args.caption_path
  
    with open(extracted_relations, 'r') as f:
        extracted_relations = [json.loads(l) for l in f.readlines()]
    outpath = args.output_file_path
    
    if os.path.exists(outpath):
        os.remove(outpath)
    for relation in extracted_relations:
        image_id = relation['image_id']
        if image_id in gt_words.keys():
            gt = gt_words[image_id]
            generated = relation['response']
            out = {
                'image_id': image_id,
                'gt_words': gt,
                'generated_words': generated,
                "generated_caption": relation['generated_caption']
            }
            with open(outpath, "a") as f:
                f.write(json.dumps(out)+"\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="objects extracting")
    parser.add_argument("-ip", "--caption_path", type=str, required=True)
    parser.add_argument("-op", "--output_file_path", type=str, required=True)
    parser.add_argument("-gt", "--gt_words", type=str,required=True)
    args = parser.parse_args()
    main(args)
