import os
import glob
import random
import json
import openai
import time
import json
import numpy as np
import os

def set_key():
    random.seed(1234)
    openai.api_key = " " ## Add your OpenAI API key here 

def llm(prompt, stop=["\n"]):
    
    success = False
    while not success:
        try:
            response = openai.ChatCompletion.create(
                model= "gpt-4",
                messages=prompt,
                temperature=0,
                top_p=1,
                frequency_penalty=0.0,
                max_tokens=2000,
                presence_penalty=0.0,
            )
            output = json.loads(response["choices"][0]["message"]["content"])
            success = True
        except Exception as e:
            print(f"Exception: {e}")
            print("Retrying...")
            time.sleep(10) 
    time.sleep(1)
    return output