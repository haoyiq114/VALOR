import json

# TODO
matched_att_obj_file_path = ""
output_file_path = ""

with open(matched_att_obj_file_path, "r") as f:
    matched_att_obj_file = [json.loads(l) for l in f.readlines()]

for matched_att_obj in matched_att_obj_file:
    # Original list of attributes from matched objects
    gt_obj_atts_list = list(matched_att_obj["gt_obj_atts"].values())

    # Splitting each string on "),", then flattening the list and stripping spaces
    gt_obj_atts_list = [item.strip() for atts in gt_obj_atts_list for item in atts.split("),")]

    # Initialize a new list for processed attributes
    new_gt_obj_atts_list = []

    # Iterate over the list and pair the attributes correctly
    for idx, item in enumerate(gt_obj_atts_list):
        # Append a ")" to odd indexed items if missing
        if idx % 2 == 0:  # Even index, potentially a start of a new tuple
            if item[-1] != ")":
                item += ")"
            new_gt_obj_atts_list.append(item)
        else:  # Odd index, simply append the item
            new_gt_obj_atts_list.append(item)
    
    # Initialize the new dictionaries
    new_matched_att_obj = {}
    new_broader_concept = {}

    # Update "new_matched_att_obj" by filtering out the relevant mappings
    for index, mapping in matched_att_obj["matched_att_obj"].items():
        matched_gt_att_obj = next(iter(mapping.values()), None)
        if matched_gt_att_obj in new_gt_obj_atts_list:
            new_matched_att_obj[index] = mapping
            new_gt_obj_atts_list.remove(matched_gt_att_obj)

    # Update "new_broader_concept" in a similar way
    for index, mapping in matched_att_obj["broader_concept"].items():
        matched_gt_att_obj = next(iter(mapping.values()), None)
        if matched_gt_att_obj in new_gt_obj_atts_list:
            mapping_str = str(mapping)
            if "unknown" and "unspecified" not in mapping_str and matched_gt_att_obj in new_gt_obj_atts_list:
                new_broader_concept[index] = mapping
                new_gt_obj_atts_list.remove(matched_gt_att_obj)
    
    current_output = {
        "image_id": matched_att_obj["image_id"],
        "gt_obj_atts": matched_att_obj["gt_obj_atts"],
        "generated_obj_atts": matched_att_obj["generated_obj_atts"],
        "matched_att_obj": new_matched_att_obj,
        "broader_concept": new_broader_concept
    }
    
    with open(output_file_path, "a") as f:
        f.write(json.dumps(current_output) + "\n")