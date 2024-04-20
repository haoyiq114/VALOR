import json
import tqdm
import sys
sys.path.append("../")
from gpt_model import llm, set_key
import argparse

def main(args):
    set_key()

    with open(args.caption_path, "r") as f:  
        generated_caps = json.load(f)
    
    prompt_template = """
        Given an image with a caption that is generated by a vision language model. 
        Please act as a linguistic master and extract all the objects from the captions and format your response in a 
        JSON format with the key being "objects" and the value being a list of objects. 
        
        Please only extract objects without including attributes. For example, extract "field" instead of "grassy field". 
        Also be mindful of plural forms. For example, extract "cow" instead of "cows".
        Don't extract the role of people, such as spectator, player, audience, mother, etc. 
        Please only extract the object that is a concrete entity in the real world instead of abstract concepts, actions, and moves. 
        It cannot be an abstract notion such as day, time, scene, moment, image, game, sport, setting, plot, atmosphere, surroundings, group etc.
        It cannot be any words describing the emotions such as excitement, enthusiasm, etc. 
        It cannot be any words describing the positions in the image, such as foreground, background, left, right, etc.

        For clarity, consider these examples:
        
        ### Example 1:
        - Caption: In the image, a young boy is holding an umbrella and petting a cow in a grassy field. 
        There are several other people scattered throughout the scene, some of whom appear to be watching the boy interacting with the cow. 
        Additionally, there are several other cows in the field, some of which are closer to the boy while others are further away. 
        The overall atmosphere is peaceful and relaxed, with the boy and the cows seemingly enjoying each other's company.
        - objects: [boy, umbrella, cow, field, people]

        ### Example 2:
        - Caption: The image depicts a group of young men playing a game of rugby on a lush green field. 
        They are gathered in the middle of the field, with one player standing on top of another's shoulders. 
        There are several other players scattered around the field, creating a lively and energetic atmosphere.  
        In addition to the rugby players, there are several other people visible in the background, possibly spectators or supporters of the game. 
        One person is positioned towards the left side of the image, while another can be seen towards the right side. 
        A third person is located closer to the center of the image. 
        Overall, the scene captures the excitement and enthusiasm of the rugby players as they engage in their sport.
        - objects: [men, field, people, person]
        
        ### Example 3: 
        - Caption: The image features a woman sitting on a green bench in front of a large church. 
        She is wearing a hat and appears to be enjoying the sunny day. She is also holding a box and enjoying her food.
        There are several other people in the scene, including a man and a child, who are walking around the area. 
        In the background, there is a fountain with water flowing from it, adding to the serene atmosphere of the scene. 
        A clock can be seen hanging on the side of the church, indicating the time of day. 
        Overall, the image captures a peaceful moment in a beautiful outdoor setting, with the green bench providing a comfortable spot for the woman to rest and enjoy the surroundings.
        - objects: [woman, bench, church, hat, box, food, people, man, child, fountain, water, clock]
        
        With these examples in mind, please help me extract the objects based on the factual information in the caption.
        Here is the caption:
        {}
    """

    outpath = args.output_file_path
    outfile = {}

    messages = [
        {
            "role": "system", 
            "content": "You are a language assistant that helps to extract information from given sentences." 
        },
    ]
    #cap = caption_ers
    for image_id, cap in tqdm.tqdm(list(generated_caps.items())):
        cap_to_gpt = cap['generated_caption']

        content = prompt_template.format(cap_to_gpt)      
        prompt = messages + [{"role": "user", "content": content}]
        llm_output = llm(prompt)
        print(llm_output)
        outfile[image_id] = llm_output
        current_output = {
            "image_id": image_id,
            "response": llm_output,
            "caption": cap_to_gpt,
        }
        with open(outpath, "a") as f:
            f.write(json.dumps(current_output)+"\n")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="objects extracting")
    parser.add_argument("-ip", "--caption_path", type=str, required=True)
    parser.add_argument("-op", "--output_file_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
