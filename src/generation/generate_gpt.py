"""
Role-play with restricted and non-restricted prompts using GPT models
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.constant import QUESTION_TYPE, TYPE
import openai  # Replace anthropic with openai
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--question_type", type=str, default="four_president")
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--cot", type=bool, default=False)
parser.add_argument("--model_id", type=str, default="gpt")  # Default to GPT-4
parser.add_argument("--exp", type=str, default="exp")

args = parser.parse_args()
torch.manual_seed(42)
random.seed(42)

# openAI initialization
openai.organization = "YOUR ORGANIZATION"
openai.api_version = "YOUR API VERSION"
openai.api_key = "YOUR API KEY"


STOP_AFTER_ATTEMPT=10

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))
def instruct_model(model_id, prompt, type):
    # Initialize client with API key from environment variable
    
    die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime or within their story. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. Please only output the answer to the questions"
    normal = "You should play the role given to you. Please only output the answer to the questions."
    normal_QA = "You should play the role given to you. Please only output the option that is correct without any other words."
    cot = " You should play the role given to you. Please think step by step and output the answer to the questions."

    die_yearly = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. Limit your knowledge to information available up to the persona's death year. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly. Please only output the answer to the questions."
    
    die_yearly_cot = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. Limit your knowledge to information available up to the persona's death year. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly.  Please think step by step and output the answer to the questions."
    if type == "die_yearly":
        system_prompt = die_yearly
    elif "normal_temp" in type:
        system_prompt = normal
    elif type == "die":
        system_prompt = die
    elif type == "cot":
        system_prompt = cot
    elif type == "die_yearly_cot":
        system_prompt = die_yearly_cot
    elif type == "normal_QA":
        system_prompt = normal_QA
    response = openai.chat.completions.create(
        model='gpt-4.1-nano-2025-04-14', # technically, you can also set the engine value as 'gpt-4', but plz don't do it!!! (to save some money lol)
        
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        temperature=.0000000000000000000001,
        # temperature=0.6,
        top_p=.0000000000000000000001,   
        max_tokens=150,     # change this when needed. 800 is just a placeholder here.
        frequency_penalty=0,
        presence_penalty=0,
        stop=["# END"]  # change this to the correct stop token based on the prompt. This stop token will only be effective if you say something like "Generate # END at the end of your response" in your prompt. Or alternatively, you can simply leave it blank.
        )
    
    answer = response.choices[0].message.content.strip()  # Extract and clean the response
    return answer

#four_president, yearly, timedied
cfg = {
    "type": args.type,
    "question_type": args.question_type,
}

if __name__ == "__main__":
    prompt_type = cfg["type"]
    question_type = cfg["question_type"]
    if question_type not in QUESTION_TYPE:
        raise ValueError("Invalid question type")
    if prompt_type not in TYPE:
        raise ValueError("Invalid type")
    
    model_id = args.model_id
    if args.exp == "exp":
        with open("answers/question.json", "r") as f:
            data = json.load(f)
    else:
        with open("answers/full_question.json", "r") as f:
            data = json.load(f)
    
    # Create a list to store all character data
    all_characters = []

    #load commonsense_cot_partial_raw_new.json
    with open("datasets/commonsense_cot_partial_raw_new.json", "r") as f:
        commonsense_data = json.load(f)
    
    # Process all categories and characters
    for category in data:
        category_type = category["type"]
        print(category_type)
        characters = category["characters"]
        
        # Skip certain categories if needed
        if category_type in ["description_human"]:
            continue
        if category_type == "fully_fictional" or category_type == "mix_fictional":
                continue
        # Process characters in each category
        for char_data in characters:
            char = char_data["role"]
            
            print(f"Character: {char} (Type: {category_type})")
            print("\n")
            
            # Construct prompt based on available data
            if "description" in char_data:
                # For characters with explicit descriptions
                prompt = f"{char_data['description']}"
            else:
                # For characters without explicit descriptions
                if category_type in ['mix_fictional', 'fully_fictional', "half_fictional", "mostly_fictional"]:
                    prompt = f"You are {char} in {char_data['source']}."
                elif category_type == "real_person":
                    prompt = f"You are {char}."
            
            # Collect all answers for this character
            all_answers = []
            if question_type == "four_president" or question_type == "four_president_cot":
                if "cot" in question_type:
                    raise ValueError("COT is not supported")
                for i in range(27):
                    question = f" Who is the {46-i}th US president?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt,prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")

            elif question_type == "commonsense":
                for i in range(len(commonsense_data)):
                    question = commonsense_data[i]["question"]
                    new_prompt = prompt + " " + question
                    print(new_prompt)
                    output = instruct_model(model_id, new_prompt,prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} {output}")

            elif question_type == "yearly" or question_type == "yearly_cot":
                if "cot" in question_type:
                    raise ValueError("COT is not supported")
                for i in range(30, -1, -1):  # Descending order from 11 to 0
                    question = f"Who is the US president in {char_data['year']+i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")

                for i in range(1, 31):  # Descending order from 11 to 1
                    question = f"Who is the US president in {char_data['year']-i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")
                
            elif question_type == "four_president_given_death":
                for i in range(27):
                    question = f"You died in {char_data['year']}. Who is the {46-i}th US president?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")

            elif question_type == "four_president_given_current":
                for i in range(27):
                    question = f"This year is 2023. Who is the {46-i}th US president?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")

            elif question_type == "four_president_given_death_current":
                for i in range(27):
                    question = f"You died in {char_data['year']}. This year is 2023. Who is the {46-i}th US president?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")
                    
            elif question_type == "yearly_given_death":
                for i in range(30, -1, -1):  # Descending order from 11 to 0
                    question = f"You died in {char_data['year']}. Who is the US president in {char_data['year']+i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")

                for i in range(1, 31):  # Descending order from 11 to 1
                    question = f"You died in {char_data['year']}. Who is the US president in {char_data['year']-i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")
            
            elif question_type == "yearly_given_death_current":
                for i in range(30, -1, -1):  # Descending order from 11 to 0
                    question = f"You died in {char_data['year']}. This year is 2023. Who is the US president in {char_data['year']+i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")

                for i in range(1, 31):  # Descending order from 11 to 1
                    question = f"You died in {char_data['year']}. This year is 2023. Who is the US president in {char_data['year']-i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")

            elif question_type == "direct_death":
                question = f"Are you dead or alive?"
                new_prompt = prompt + f" {question} Answer:"
                print(question)
                output = instruct_model(model_id, new_prompt, prompt_type)
                print(output)
                print("\n")
                all_answers.append(f"Question: {question}\nAnswer: {output}\n")

            elif question_type == "direct_death_current":
                question = f"This year is 2023. Are you dead or alive?"
                new_prompt = prompt + f" {question} Answer:"
                print(question)
                output = instruct_model(model_id, new_prompt, prompt_type)
                print(output)
                print("\n")
                all_answers.append(f"Question: {question}\nAnswer: {output}\n")
            
            elif question_type == "direct_year_current_death":
                question = f"You died in {char_data['year']}. This year is 2023. Are you dead or alive?"
                new_prompt = prompt + f" {question} Answer:"
                print(question)
                output = instruct_model(model_id, new_prompt, prompt_type)
                print(output)
                print("\n")
                all_answers.append(f"Question: {question} Answer: {output}")
            
            elif question_type == "when_die":
                question = f"Which year did you die?"
                new_prompt = prompt + f" {question} Answer:"
                print(question)
                output = instruct_model(model_id, new_prompt, prompt_type)
                print(output)
                print("\n")
                all_answers.append(f"Question: {question}\nAnswer: {output}\n")

            elif question_type == "direct_death_year":
                for i in range(30, -1, -1):  # Descending order from 11 to 0
                    question = f"Are you dead or alive in {char_data['year']+i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")

                for i in range(1, 31):  # Descending order from 11 to 1
                    question = f"Are you dead or alive in {char_data['year']-i}?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(question)
                    output = instruct_model(model_id, new_prompt, prompt_type)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question}\nAnswer: {output}\n")
        
            # Create character entry for JSON
            character_entry = {
                "character": char,
                "type": category_type,
                "questions": all_answers
            }
            
            # Add death_year if available
            if "death_year" in char_data:
                character_entry["death_year"] = char_data["death_year"]
                
            all_characters.append(character_entry)
            
            # Write character data to JSON file after processing each character
            os.makedirs(f"answers/{args.exp}/{model_id}/{prompt_type}", exist_ok=True)
            with open(f"answers/{args.exp}/{model_id}/{prompt_type}/formatted_output_{prompt_type}_{question_type}.json", "w") as f:
                json.dump(all_characters, f, indent=4)
    
    print(f"answers/{args.exp}/{model_id}/{prompt_type}/formatted_output_{prompt_type}_{question_type}.json")