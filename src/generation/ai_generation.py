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
from transformers import pipeline
from tqdm import tqdm
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

if args.question_type not in QUESTION_TYPE:
    raise ValueError("Invalid question type")
if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.model_id not in ["llama", "gemma", "claude", 'gpt']:
    raise ValueError("Invalid model id")


def instruct_model(model_id, prompt):
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
        device="cuda",  # replace with "mps" to run on a Mac device
        )
    else:
        raise ValueError("Invalid model id")
    die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime or within their story. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. Please only output the answer to the questions"
    normal = "You should play the role given to you. Please only output the answer to the questions."
    
    ai = "Please think step by step and carefully consider the context and only output the answer to the questions."
    ai_QA = "Please think step by step and carefully consider the context and only output the option that is correct without any other words."
    if "llama" in model_id:
        messages = [
                {"role": "system", "content": ai_QA},
                {"role": "user", "content": prompt}
        ]
    elif "gemma" in model_id:
        messages = [
            {"role": "user", "content": ai + " " + prompt}
        ]
    outputs = pipe(
        messages,
        max_new_tokens=100,
        temperature=0.6,
    )
    return outputs[0]["generated_text"][-1]['content']

cfg = {
    "type": args.type,
    "question_type": args.question_type
}

if __name__ == "__main__":
    model_id = args.model_id
    all_answers = []
    with open("answers/full_question.json", "r") as f:
        data = json.load(f)

        # breakpoint()
     #load commonsense_cot_partial_raw_new.json
    with open("datasets/commonsense_cot_partial_raw_new.json", "r") as f:
        commonsense_data = json.load(f)

    if cfg["question_type"] == "when_die":
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
                    question = f"Which year did {char} die?" 

                    output = instruct_model(model_id, question)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")

    elif cfg["question_type"] == "commonsense":
                for i in range(len(commonsense_data)):
                    question = commonsense_data[i]["question"]
                    print(question)
                    output = instruct_model(model_id, question)
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

                    output = instruct_model(model_id, question)
                    print(output)
                    print("\n")
                    all_answers.append(f"Question: {question} Answer: {output}")
    elif cfg["question_type"] == "four_president":
        for i in range(27):
            question = f"Who is the {46-i}th US president?"
            prompt = f"{question}"
            output  = instruct_model(model_id, prompt)
            print(output)
            print("\n")

    elif cfg["question_type"] == "yearly" or cfg["question_type"] == "yearly_cot":
        if "cot" in cfg["question_type"]:
            raise ValueError("COT is not supported")
        for i in range(1881,2022):  # Descending order from 11 to 0
            question = f"Who is the US president in {i}?"
            new_prompt = f"{question} Answer:"
            print(question)
            output = instruct_model(model_id, new_prompt)
            print(output)
            print("\n")
            all_answers.append(f"Question: {question}\nAnswer: {output}\n")
    
    character_entry = {
        "character": 'AI',
        "type": cfg["question_type"],
        "questions": all_answers
    }

    os.makedirs(f"answers/ai_temp3/{model_id}", exist_ok=True)
    with open(f"answers/ai_temp3/{model_id}/formatted_output_{cfg['type']}_{cfg['question_type']}.json", "w") as f:
        json.dump([character_entry], f, indent=4)