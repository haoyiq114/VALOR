import json
import argparse
import os

class CHAIR(object):

    def __init__(self, args):
        
        with open(args.matched_objects_file_path, "r") as f:
            self.matched_objects_file = [json.loads(l) for l in f.readlines()]
            
        self.imageids = []
        self.imageid_to_captions = {}
        self.imageid_to_gt_objects = {}
        self.imageid_to_ge_objects = {}
        self.imageid_to_matched_objects = {}
        self.imageid_to_mapping = {}
        self.imageid_to_broader_concept = {}
        
        for image_info in self.matched_objects_file:
            image_id = image_info["image_id"]
          
            self.imageids.append(image_id)
            self.imageid_to_captions[image_id] = image_info["caption"]
            self.imageid_to_gt_objects[image_id] = list(image_info["gt_words"])
            self.imageid_to_ge_objects[image_id] = image_info["generated_words"]
            self.imageid_to_matched_objects[image_id] = image_info["matched_objects"]
            # self.imageid_to_mapping[image_id] = image_info["mapping"]
            self.imageid_to_broader_concept[image_id] = image_info["broader_concept"]

    def compute_chair(self):
        
        imageids = self.imageids
 
        num_caps = 0.
        num_faithful_caps = 0.
        num_covered_caps = 0.
        faithful_word_count = 0.
        covered_word_count = 0.
        ge_word_count = 0.
        gt_word_count = 0.

        output = {"sentences": []} 
    
        for image_id in imageids:

            gt_objects = self.imageid_to_gt_objects[image_id]
            ge_objects = self.imageid_to_ge_objects[image_id]
            matched_objects = list(self.imageid_to_matched_objects[image_id].values())
            broader_concepts = self.imageid_to_broader_concept[image_id]
            caption = self.imageid_to_captions[image_id]

            cap_dict = {
                "image_id": image_id, 
                "caption": caption,
                "gt_words": list(gt_objects),
                "generated_words": list(ge_objects),
                "matched_words": list(matched_objects),
                "metrics": {
                    "faithfulness_score_s": 0,
                    "faithfulness_score_i": 0,
                    "coverage_score_i": 0,
                    "coverage_score_s": 0,
                    "overall_i": 0
                }
            }
 
            # count faithful words
            ge_word_count += len(ge_objects)
            faithful_word_len = len(matched_objects) + len(set(broader_concepts.keys()))
            faithful_word_count += faithful_word_len
            faithful = faithful_word_len == len(ge_objects)

            # count covered words
            gt_word_count += len(set(gt_objects))
            covered_word_count += len(set(matched_objects))          
            covered = len(set(matched_objects)) == len(set(gt_objects))
    
            # count faithful caps
            num_caps += 1
            if faithful:
               num_faithful_caps += 1

            # count covered caps
            if covered:
               num_covered_caps += 1
    
            cap_dict["metrics"]["faithfulness_score_s"] = int(faithful)
            cap_dict["metrics"]["faithfulness_score_i"] = faithful_word_len / float(len(ge_objects)) if len(ge_objects) > 0 else 0.
            cap_dict["metrics"]["coverage_score_s"] = int(covered)
            cap_dict["metrics"]["coverage_score_i"] = len(set(matched_objects)) / float(len(set(gt_objects))) if len(gt_objects) > 0 else 0.
            cap_dict["metrics"]["overall_i"] = cap_dict["metrics"]["faithfulness_score_i"] + cap_dict["metrics"]["coverage_score_i"]

            output["sentences"].append(cap_dict)
 
        faithfulness_score_s = num_faithful_caps / num_caps
        faithfulness_score_i = faithful_word_count / ge_word_count

        coverage_score_s = num_covered_caps / num_caps
        coverage_score_i = covered_word_count / gt_word_count

        output["overall_metrics"] = {
            "faithfulness_score_s": faithfulness_score_s,
            "faithfulness_score_i": faithfulness_score_i,
            "coverage_score_s": coverage_score_s,
            "coverage_score_i": coverage_score_i
        }

        return output 

def save_evaluation_results(args): 
    tag = args.matched_objects_file_path.split("/")[-1].split(".")[0]
    
    if not os.path.exists(args.evaluation_results_path):
        os.makedirs(args.evaluation_results_path)
    
    save_file_path = f"{args.evaluation_results_path}/chair_evaluation_results_{tag}.json"
    
    with open(save_file_path, "w") as f:
        json.dump(cap_dict, f, indent=4)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict["overall_metrics"]
    metric_string = "%0.01f\t\t\t%0.01f\t\t\t%0.01f\t\t\t%0.01f" %(sentence_metrics["faithfulness_score_s"]*100,
                                                           sentence_metrics["faithfulness_score_i"]*100,
                                                           sentence_metrics["coverage_score_s"]*100,
                                                           sentence_metrics["coverage_score_i"]*100
                                                          )

    if not quiet:
        print("faithfulness_score_s\tfaithfulness_score_i\tcoverage_score_s\tcoverage_score_i")
        print(metric_string)

    else:
        return metric_string
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--matched_objects_file_path", type=str, required=True) #extract_with_llm/tmp/objects_gqa/matching/human_50
    parser.add_argument("-o","--evaluation_results_path", type=str,required=True) #extract_with_llm/tmp/objects_gqa/matching/human_50
    args = parser.parse_args()

    evaluator = CHAIR(args) 
    cap_dict = evaluator.compute_chair() 
    
    print_metrics(cap_dict)
    save_evaluation_results(args)