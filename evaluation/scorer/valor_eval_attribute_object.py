import json
import argparse
import os

class CHAIR(object):

    def __init__(self, args):
        
        with open(args.matched_obj_atts_file_path, "r") as f:
            self.matched_obj_atts_file = [json.loads(l) for l in f.readlines()]
        
        self.imageids = []
        self.imageid_to_captions = {}
        self.imageid_to_gt_obj_atts = {}
        self.imageid_to_ge_obj_atts = {}
        self.imageid_to_matched_obj_atts = {}
        self.imageid_to_broader_concept = {}

        for image_info in self.matched_obj_atts_file:
            image_id = image_info["image_id"]
            self.imageids.append(image_id)
            self.imageid_to_gt_obj_atts[image_id] = image_info["gt_obj_atts"]
            self.imageid_to_ge_obj_atts[image_id] = image_info["generated_obj_atts"]
            self.imageid_to_matched_obj_atts[image_id] = image_info["matched_att_obj"]
            self.imageid_to_broader_concept[image_id] = image_info["broader_concept"]

    def compute_chair(self):
        
        imageids = self.imageids
 
        num_caps = 0.
        num_faithful_caps = 0.
        num_covered_caps = 0.
        faithful_obj_atts_count = 0.
        covered_obj_atts_count = 0.
        ge_obj_atts_count = 0.
        gt_obj_atts_count = 0.

        output = {"sentences": []} 
    
        for image_id in imageids:

            gt_obj_atts = self.imageid_to_gt_obj_atts[image_id].values()
            ge_obj_atts = self.imageid_to_ge_obj_atts[image_id].values()
            matched_obj_atts = self.imageid_to_matched_obj_atts[image_id].values()
            broader_concepts = self.imageid_to_broader_concept[image_id].values()
            matched_obj_atts = [list(matched_obj_att.values())[0] for matched_obj_att in matched_obj_atts]
            broader_concepts = [list(broader_concept.values())[0] for broader_concept in broader_concepts]

            cap_dict = {
                "image_id": image_id, 
                "gt_obj_atts": list(gt_obj_atts),
                "generated_obj_atts": list(ge_obj_atts),
                "matched_obj_atts": matched_obj_atts,
                "broader_concept": broader_concepts,
                "metrics": {
                    "faithfulness_score_s": 0,
                    "faithfulness_score_i": 0,
                    "coverage_score_i": 0,
                    "coverage_score_s": 0,
                    "overall_i": 0
                }
            }
 
            # count faithful words
            ge_obj_atts_count += len(ge_obj_atts)
            faithful_obj_atts_len = len(matched_obj_atts) + len(broader_concepts)
            faithful_obj_atts_count += faithful_obj_atts_len
            faithful = faithful_obj_atts_len == len(ge_obj_atts)

            # count covered words
            gt_obj_atts_count += len(gt_obj_atts)
            covered_obj_atts_count += len(matched_obj_atts)         
            covered = len(matched_obj_atts) == len(gt_obj_atts)
    
            # count faithful caps
            num_caps += 1
            if faithful:
               num_faithful_caps += 1

            # count covered caps
            if covered:
               num_covered_caps += 1
    
            cap_dict["metrics"]["faithfulness_score_s"] = int(faithful)
            cap_dict["metrics"]["faithfulness_score_i"] = faithful_obj_atts_len / float(len(ge_obj_atts)) if len(ge_obj_atts) > 0 else 0.
            cap_dict["metrics"]["coverage_score_s"] = int(covered)
            cap_dict["metrics"]["coverage_score_i"] = len(matched_obj_atts) / float(len(gt_obj_atts)) if len(gt_obj_atts) > 0 else 0.
            cap_dict["metrics"]["overall_i"] = cap_dict["metrics"]["faithfulness_score_i"] + cap_dict["metrics"]["coverage_score_i"]

            output["sentences"].append(cap_dict)
 
        faithfulness_score_s = num_faithful_caps / num_caps
        faithfulness_score_i = faithful_obj_atts_count / ge_obj_atts_count

        coverage_score_s = num_covered_caps / num_caps
        coverage_score_i = covered_obj_atts_count / gt_obj_atts_count

        output["overall_metrics"] = {
            "faithfulness_score_s": faithfulness_score_s,
            "faithfulness_score_i": faithfulness_score_i,
            "coverage_score_s": coverage_score_s,
            "coverage_score_i": coverage_score_i
        }

        return output 

def save_evaluation_results(args): 
    tag = args.matched_obj_atts_file_path.split("/")[-1].split(".")[0]
    
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
    parser.add_argument("-mp", "--matched_obj_atts_file_path", type=str, required=True)
    parser.add_argument("-ep", "--evaluation_results_path", type=str, required=True)
    
    args = parser.parse_args()

    evaluator = CHAIR(args) 
    cap_dict = evaluator.compute_chair() 
    
    print_metrics(cap_dict)
    save_evaluation_results(args)