import json
import os
import numpy as np
import argparse
from copy import deepcopy

def longest_monotonic_subsequence(arr):
    """
    Find the longest monotonic increasing subsequence in an array.

    :param arr: List of integers (ranks)
    :return: The longest monotonic increasing subsequence.
    """
    if not arr:
        return []

    # Initialize the dp array where dp[i] will store the longest increasing subsequence ending with arr[i]
    dp = [[num] for num in arr]

    # Fill the dp array
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] > arr[j] and len(dp[i]) < len(dp[j]) + 1:
                dp[i] = dp[j] + [arr[i]]

    # Find the longest subsequence
    longest_subsequence = max(dp, key=len)

    return longest_subsequence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ranked objects matching")
    parser.add_argument("-ip", "--input_file_path", type=str, required=True) #expect jsonl format
    parser.add_argument("-op", "--output_file_path", type=str, required=True) #expect jsonl format
    args = parser.parse_args()
    
    with open(args.input_file_path, "r") as f:
        cap_info = [json.loads(l) for l in f.readlines()]
    
    for item in cap_info:
        
        if item["generated_words"] == "":
            item["matched_objects_filtered"] = []
            item["num_broader_concept_filtered"] = 0
            continue
        
        gt_to_rank = {}
        for k,v in item["gt_words"].items():
            gt_to_rank[v] = k
      
        broader_concept = item["broader_concept"]
        broader_concept_copy = deepcopy(broader_concept)
  
        for k,o in broader_concept_copy.items():
            if o in gt_to_rank.keys():
                pass
            else:
                for oo in o.split('/'):
                    oo = oo.strip()
                    if oo in gt_to_rank.keys():
                        broader_concept.pop(k, o)
                        broader_concept[k] = oo
        broader_concept_ranks = [int(gt_to_rank[o]) for o in broader_concept.values()]
        item["broader_concept"] = broader_concept
        joint_objects = {**item["broader_concept"], **item["matched_objects"]}
       
        mapped_objects = []
        for k,v in item['generated_words'].items():
            if v in joint_objects.keys():
                mapped_objects.append(joint_objects[v])

        ranks =[int(gt_to_rank[o])for o in mapped_objects]
       
        ranks= longest_monotonic_subsequence(ranks)
      
        item["matched_objects_filtered"] = [item['gt_words'][str(i)] for i in ranks]
        
        diff = len(set(broader_concept_ranks) - set(ranks))
        item["num_broader_concept_filtered"] = diff
    
    with open(args.output_file_path, "w") as f:
        for item in cap_info:
            f.write(json.dumps(item) + "\n")
        


    