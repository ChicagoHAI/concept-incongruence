"""
Non-role-play setting generation: 
already have all the baseline
"""
import torch
import os
import json
import random
import argparse
import sys
import openai  # Add this import
from transformers import pipeline
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from pprint import pprint

from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.constant import QUESTION_TYPE, TYPE


torch.manual_seed(42)
random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--question_type", help="question type only from [four_president, four_president_given_death, four_president_given_death_current, yearly, yearly_given_death, direct_death, direct_death_current, when_die, direct_death_year]", type=str, default="four_president")
parser.add_argument("--type", help="type only from [normal, die]", type=str, default="normal")
parser.add_argument("--model_id", help="model name only from [llama, gemma, claude, gpt]", type=str, default="llama")
args = parser.parse_args()


# openAI initialization
openai.organization = "YOUR ORGANIZATION"
openai.api_version = "YOUR API VERSION"
openai.api_key = "YOUR API KEY"


STOP_AFTER_ATTEMPT=10

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))

def instruct_model( prompt):
    # Initialize client with API key directly in the code
    
    die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime or within their story. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. Please only output the answer to the questions"
    normal = "You should play the role given to you. Please only output the answer to the questions."
    ai = "Please think step by step and carefully consider the context and only output the answer to the questions."
    cot = " You should play the role given to you. Please think step by step and output the answer to the questions."
    
    ai_QA = "Please think step by step and carefully consider the context and only output the option that is correct without any other words."
    

    system_prompt = ai_QA
    
    response = openai.chat.completions.create(
        model='gpt-4.1-nano-2025-04-14',
        
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        temperature=.0000000000000000000001,
        temperature=0.6,
        top_p=.0000000000000000000001,   
        max_tokens=150,  
        frequency_penalty=0,
        presence_penalty=0,
        stop=["# END"] 
        )
    
    answer = response.choices[0].message.content.strip()  # Extract and clean the response
    return answer

cfg = {
    "type": args.type,
    "question_type": args.question_type
}

if __name__ == "__main__":
    all_answers = []
    with open("answers/full_question.json", "r") as f:
        data = json.load(f)

    #load commonsense_cot_partial_raw_new.json
    with open("datasets/commonsense_cot_partial_raw_new.json", "r") as f:
        commonsense_data = json.load(f)

    if cfg["question_type"] == "when_die":
            for category in data:
                category_type = category["type"]
                characters = category["characters"]
                # Skip certain categories if needed
                if category_type in ["description_human", "fully_fictional"]:
                    continue

                for char_data in characters:
                    char = char_data["role"]
                    death_year = char_data["year"]
                    source = char_data["source"]
                    question = f"Which year did {char} die?" 

                    output = instruct_model(question)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")

    elif cfg["question_type"] == "commonsense":
                for i in range(len(commonsense_data)):
                    question = commonsense_data[i]["question"]
                    print(question)
                    output = instruct_model(question)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} {output}")

    elif cfg["question_type"] == "direct_death":
        for category in data:
                category_type = category["type"]
                characters = category["characters"]
                # Skip certain categories if needed
                if category_type in ["description_human", "fully_fictional", "mix_fictional"]:
                    continue

                for char_data in characters:
                    char = char_data["role"]
                    death_year = char_data["year"]
                    source = char_data["source"]
                    question = f"Was {char} dead or alive?" 

                    output = instruct_model(question)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")
    elif cfg["question_type"] == "four_president":
        for i in range(27):
            question = f"Who is the {46-i}th US president?"
            prompt = f"{question}"
            output  = instruct_model(prompt)
            print(output)
            print("\n")
            all_answers.append(f"Question: {question} Answer: {output}")

    elif cfg["question_type"] == "yearly" or cfg["question_type"] == "yearly_cot":
        if "cot" in cfg["question_type"]:
            raise ValueError("COT is not supported")
        for i in range(1881,2022):  # Descending order from 11 to 0
            question = f"Who is the US president in {i}?"
            new_prompt = f"{question} Answer:"
            print(question)
            output = instruct_model(new_prompt)
            print(output)
            print("\n")
            all_answers.append(f"Question: {question}\nAnswer: {output}\n")
    
    character_entry = {
        "character": 'AI',
        "type": cfg["question_type"],
        "questions": all_answers
    }

    os.makedirs(f"answers/{cfg['type']}/gpt", exist_ok=True)
    with open(f"answers/{cfg['type']}/gpt/formatted_output_{cfg['type']}_{cfg['question_type']}.json", "w") as f:
        json.dump([character_entry], f, indent=4)
