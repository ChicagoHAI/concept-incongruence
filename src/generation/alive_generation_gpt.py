"""
Role-play with alive characters
"""

import torch
import os
import json
import random
from transformers import pipeline
from tqdm import tqdm
from pprint import pprint
import argparse
import sys
import openai
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.constant import TYPE, QUESTION_TYPE
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--question_type", type=str, default="four_president")
parser.add_argument("--type", type=str, default="yearly")
parser.add_argument("--model_id", type=str, default="llama")
args = parser.parse_args()



torch.manual_seed(42)
random.seed(42)
live_character = ["Taylor Swift", "Justin Bieber", "Elon Musk", "Emma Stone", "Tom Cruise", "Beyonce"]

# openAI initialization
openai.organization = "YOUR ORGANIZATION"
openai.api_version = "YOUR API VERSION"
openai.api_key = "YOUR API KEY"


STOP_AFTER_ATTEMPT=10

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))

def instruct_model(model_id, prompt):
    die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime or within their story. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. Please only output the answer to the questions"
    normal = "You should play the role given to you. Please only output the answer to the questions."
    cot = "Please think step by step and output the answer to the questions."
    
    system_prompt = normal
    response = openai.chat.completions.create(
        model='gpt-4.1-nano-2025-04-14',
        
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        temperature=.0000000000000000000001,
        top_p=.0000000000000000000001,   
        max_tokens=150,     
        frequency_penalty=0,
        presence_penalty=0,
        stop=["# END"]  
        )
    
    answer = response.choices[0].message.content.strip()  # Extract and clean the response
    return answer



#four_president, yearly, timedied
cfg = {
    "type": "alive",
    "question_type": "four_president",
    "model_id": "gpt"
}

if __name__ == "__main__":
    type = cfg["type"]
    question_type = cfg["question_type"]
    model_id = args.model_id
    
    # Create a list to store all character data
    all_characters = []
    #FIXME: it will include all the characters
    for char in live_character:
        all_answers = []  # Reset answers list for each character
        for i in range(27):
            prompt = f"You are {char}. Who is the {46-i}th US president?"
            answer = instruct_model(model_id, prompt)
            all_answers.append(f"Question: {prompt} Answer: {answer}")
            print(f"Question: {prompt} Answer: {answer}")

        character_entry = {
            "character": char,
            "type": "alive",
            "questions": all_answers
        }
        all_characters.append(character_entry)
    
    # Get the absolute path to ensure we're saving in the right location
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(root_dir, "answers", "full", "gpt", "alive", "four_president")
    save_path = os.path.join(save_dir, "formatted_output_four_president.json")
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_characters, f, indent=4)
        print(f"Successfully saved results to: {save_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    # Update the final print to match actual save location
    print(f"File saved to: {save_path}")