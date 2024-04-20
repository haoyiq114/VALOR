import json
import argparse
from copy import deepcopy
def postprocess(image_id, objects):
    
    # Unwanted objects
    fake_objects = ["atmosphere", "scene", "sport","sports","game", "time", "trip", "items", 
                    "scenery", "foreground", "background", "group", "backdrop", "meal", 
                    "day", "days", "conversation", "protection", "sense of", "product", "appliance", 
                    "outdoor", "activity", "activities", "event", "events", "occasion", "occasions",
                    "work", "works", "job", "jobs", "occupation", "occupations","journey", 'design', 'comfort', 'elegance', 'relaxation', 'grooming'
                    ]
    positions = ["left side", "right side", "top left", "top right", "bottom left", "bottom right", "center", "middle", "top", "bottom", "front", "back"]
    # body_parts = ["head", "face", "eyes", "mouth", "nose", "ears", "hair", "neck", "shoulder", "chest", "arm", "elbow", "wrist", "hand", "fingers", "leg", "knee", "ankle", "foot", "toes"]
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    # space 
    space = ["space", "area", "room", "cityscape", "shoreline"]
    color = ["greenery", "blue", "red", "yellow", "orange", "purple", "pink", "white", "black", "grey", 
             "brown", "colorful", "color", "colors", "colored", "colours", "coloured", "colour"]
    # also check family, girl, boy this kind of words
    family = ["mother", "father", "brother", "sister", "grandmother", "grandfather", "aunt", "uncle", 
              "cousin", "niece", "nephew", "daughter", "son", "wife", "husband"]
    relationships = ["friend", "lover" ]
    scene = ["sun", "sunrays", "sunray"]
    roles = ["spectator", "spectators", "player", "audience", "mother", "supporter", "supporters", 'players', 'spectators', 'sidelines','batter', 'batter', 'catcher', 'umpire']
    new_add = ['shapes', 'sizes','access','image', 'picture', 'options', "side", "sides", "adventure", "parasailing", "gesture", 'beauty', 'surroundings'
               "moment", 'Ben', 'Westminster', 'Thames', 'context', 'setting', 'moment', "trick", 'skill', 'enthusiasm', 'place', 'break', "couple",'gathering', 
               "pitch", 'zone', 'facilities', 'coloration', 'pigments', 'diet', 'appearance', 'environment', 'bokeh', 'portrait','photograph', 'format', 'border', 'positioning', 'posture', 
               'structure', 'service', 'reflection','photography', "arrangements",'patterns','shadows','beams', 'motion', 'concentration', 'solitude',
               "sunlight", 'overpass', 'movement', 'life', 'transportation', 'nature','horizon','daylight', 'lighting', 'patterns','habitat', 'frame', 
               'landscape', 'music', 'facade', 'rides', 'shape', 'climate', 'location', 'neighborhood', 'interior']
    
    filtered_objects = []
    for obj in objects:
        #filter action word ending with "ing", might use nltk to improve this? 
        # if obj.endswith("ing"):
        #     objects.remove(obj)
        #     next
        words = obj.split()
        for word in words:
            if word in fake_objects or word in positions or word in space or word in numbers or word in color \
            or word in relationships or word in scene or word in roles or word in new_add or word in family:
                filtered_objects.append(obj)
    
    print(image_id, f"Filtered {len(filtered_objects)} objects", filtered_objects)             
    returned_objects = [obj for obj in objects if obj not in filtered_objects]

    return returned_objects

def main(args):    
    with open(args.caption_path, "r") as f:
        captions = [json.loads(l) for l in f.readlines()]
        
    for item in captions:
        image_id = item["image_id"]
        objects = item["response"]["objects"]
        objects = postprocess(image_id, objects)
        out = {
            "image_id": image_id,
            "objects": objects,
            "caption": item["caption"]
        }
        with open(args.output_path, "a") as f:
                f.write(json.dumps(out)+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess the generated captions")
    parser.add_argument("-c", "--caption_path", type=str, required=True, help="Path to the generated captions") #expect jsonl format
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save the postprocessed captions") #expect jsonl format
    args = parser.parse_args()
    main(args)
 