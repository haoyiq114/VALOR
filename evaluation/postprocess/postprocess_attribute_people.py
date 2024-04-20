import json

# TODO
matched_att_obj_file_path = ""
output_file_path = ""

with open(matched_att_obj_file_path, "r") as f:
    matched_att_obj_file = [json.loads(l) for l in f.readlines()]

for matched_att_obj in matched_att_obj_file:
    # Original list of attributes from matched objects
    gt_obj_atts = matched_att_obj["gt_obj_atts"].values()
    gt_obj_atts = [gt_obj_att["object"] for gt_obj_att in gt_obj_atts]
    # Splitting each string on "),", then flattening the list and stripping spaces
    gt_obj_atts_list = [item.strip() for atts in gt_obj_atts for item in atts.split("),")]
    # Initialize a new list for processed attributes
    new_gt_obj_atts_list = []
    for item in gt_obj_atts_list:
        # Append a ")" to odd indexed items if missing
        if item[-1] != ")":
            item += ")"
            new_gt_obj_atts_list.append(item)
    
    # Initialize the new dictionaries
    new_matched_att_obj = {}
    new_broader_concept = {}

    # Update "new_matched_att_obj" by filtering out the relevant mappings
    for index, mapping in matched_att_obj["matched_att_obj"].items():
        if mapping.get("object") != None:
            matched_obj_atts_dict = mapping.get("object")
            new_mapping = {}
            new_mapping_objects = {}
            for matched_ge_obj_atts, matched_gt_obj_atts in matched_obj_atts_dict.items():
                if matched_gt_obj_atts in new_gt_obj_atts_list:
                    new_mapping_objects[matched_ge_obj_atts] = matched_gt_obj_atts
                    new_gt_obj_atts_list.remove(matched_gt_obj_atts)
            if mapping.get("person") != None and mapping.get("object") != None:
                new_mapping["person"] = mapping["person"]
                new_mapping["object"] = new_mapping_objects
                new_matched_att_obj[index] = new_mapping

    # Update "new_broader_concept" in a similar way
    for index, mapping in matched_att_obj["broader_concept"].items():
        if mapping.get("object") != None:
            matched_obj_atts_dict = mapping.get("object")
            new_mapping = {}
            new_mapping_objects = {}
            for matched_ge_obj_atts, matched_gt_obj_atts in matched_obj_atts_dict.items():
                if matched_gt_obj_atts in new_gt_obj_atts_list:
                    new_mapping_objects[matched_ge_obj_atts] = matched_gt_obj_atts
                    new_gt_obj_atts_list.remove(matched_gt_obj_atts)
            if mapping.get("person") != None and mapping.get("object") != None:
                new_mapping["person"] = mapping["person"]
                new_mapping["object"] = new_mapping_objects
                new_broader_concept[index] = new_mapping
    
    current_output = {
        "image_id": matched_att_obj["image_id"],
        "gt_obj_atts": matched_att_obj["gt_obj_atts"],
        "generated_obj_atts": matched_att_obj["generated_obj_atts"],
        "matched_att_obj": new_matched_att_obj,
        "broader_concept": new_broader_concept
    }
    
    with open(output_file_path, "a") as f:
        f.write(json.dumps(current_output) + "\n")