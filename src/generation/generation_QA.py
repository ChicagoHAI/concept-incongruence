"""
Role-play with restricted and non-restricted prompts
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

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--question_type", help="question type only from [four_president, four_president_given_death, four_president_given_death_current, yearly, yearly_given_death, direct_death, direct_death_current, when_die, direct_death_year]", type=str, default="four_president")
parser.add_argument("--type", help="type only from [normal, die]", type=str, default="normal")
parser.add_argument("--cot", type=bool, default=False)
parser.add_argument("--model_id", help="model name only from [llama, gemma, claude, gpt]", type=str, default="llama")
parser.add_argument("--experiment_type", help="type of experiment (exp or full)", type=str, default="exp")
parser.add_argument("--continue_from_file", type=bool, default=False)
args = parser.parse_args()

if args.question_type not in QUESTION_TYPE:
    raise ValueError("Invalid question type")
if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.model_id not in ["llama", "gemma", "claude", 'gpt']:
    raise ValueError("Invalid model id")


torch.manual_seed(42)
random.seed(42)
def instruct_model(model_id, prompt, type):
    if "llama" in model_id:
        pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        temperature=0.6,
        )
    elif "gemma" in model_id:
        pipe = pipeline(
        "text-generation",
        model="google/gemma-2-9b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        temperature=0.6,
        device="cuda",  # replace with "mps" to run on a Mac device
        )
    else:
        raise ValueError("Invalid model id")
    
    die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime or within their story. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. Please only output the answer to the questions"
    normal = "You should play the role given to you. Please only output the answer to the questions."
    normal_QA = "You should play the role given to you. Please only output the option that is correct without any other words."
    cot = " You should play the role given to you. Please think step by step and output the answer to the questions."
    more_die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly. Please only output the answer to the questions"

    die_yearly = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. Limit your knowledge to information available up to the persona's death year. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly. Please only output the answer to the questions."

    die_yearly_cot = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. Limit your knowledge to information available up to the persona's death year. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly.  Please think step by step and output the answer to the questions."

    if "llama" in model_id:
        if type == "more_die":
            system_prompt = more_die
        elif type == "die":
            system_prompt = die
        elif type == "normal":
            system_prompt = normal
        elif type == "die_yearly":
            system_prompt = die_yearly
        elif type == "cot":
            system_prompt = cot
        elif type == "die_yearly_cot":
            system_prompt = die_yearly_cot
        elif type == "normal_QA":
            system_prompt = normal_QA
        print(system_prompt)
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
        ]
    elif "gemma" in model_id:
        if type == "more_die":
            system_prompt = more_die
        elif type == "die":
            system_prompt = die
        elif type == "normal":
            system_prompt = normal
        elif type == "die_yearly":
            system_prompt = die_yearly
        elif type == "cot":
            system_prompt = cot
        elif type == "normal_QA":
            system_prompt = normal_QA
        messages = [
            {"role": "user", "content": system_prompt + " " + prompt},
        ]
    outputs = pipe(
        messages,
        max_new_tokens=150,
        # do_sample=False,
        temperature=0.6
    )
    
    tokenizer = pipe.tokenizer
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return string, not token IDs
        add_generation_prompt=True  # Mimic pipeline behavior
    )
    print("Formatted Prompt:")
    print(formatted_prompt)
    return outputs[0]["generated_text"][-1]['content']



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
    if args.experiment_type == "exp":
        with open("answers/question.json", "r") as f:
            data = json.load(f)
    else:
        with open("answers/full_question.json", "r") as f:
            data = json.load(f)
    
    # Create a list to store all character data
    all_characters = []
    processed_characters = []

    #load commonsense_cot_partial_raw_new.json
    with open("datasets/commonsense_cot_partial_raw_new.json", "r") as f:
        commonsense_data = json.load(f)
    
        
    
    # Process all categories and characters
    for category in data:
        category_type = category["type"]
        print(category_type)
        characters = category["characters"]
        
        # Skip certain categories if needed
        #FIXME: skip all fully fictional
        if category_type in ["description_human"] or category_type == "fully_fictional" or category_type == "mix_fictional":
            continue
        # Process characters in each category
        #FIXME: remove this
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
                if category_type in ['mostly_fictional', 'fully_fictional','half_fictional', 'mix_fictional']:
                    prompt = f"You are {char} in {char_data['source']}."
                    # prompt = f"Act as {char} in {char_data['source']}."
                elif category_type == "real_person":
                    prompt = f"You are {char}."
                    # prompt = f"Act as {char}."
            
            print(prompt)
            # Collect all answers for this character
            all_answers = []
            if question_type == "four_president" or question_type == "four_president_cot":
                if "cot" in question_type:
                    raise ValueError("COT is not supported")
                for i in range(27):
                    question = f" Who is the {46-i}th US president?"
                    new_prompt = prompt + f" {question} Answer:"
                    print(new_prompt)
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
            
            # breakpoint()
            
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
            os.makedirs(f"answers/{args.experiment_type}/{model_id}/{prompt_type}", exist_ok=True)
            with open(f"answers/{args.experiment_type}/{model_id}/{prompt_type}/formatted_output_{prompt_type}_{question_type}.json", "w") as f:
                    json.dump(all_characters, f, indent=4)

            
    
    print(f"answers/{args.experiment_type}/{model_id}/{prompt_type}/formatted_output_{prompt_type}_{question_type}.json")
