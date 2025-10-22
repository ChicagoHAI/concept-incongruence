"""
Accuracy evaluation
"""
import numpy as np  
import openai
import json
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from src.prompt_template import ACC_EVAL_PROMPT, ACC_EVAL_PROMPT_AI
from src.constant import TYPE, QUESTION_TYPE
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="llama")
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--question_type", type=str, default="four_president_given_death_current")
parser.add_argument("--experiment_type", type=str, default="exp")
parser.add_argument("--continue_eval", action="store_true", help="Continue evaluation from previous run")

args = parser.parse_args()

if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.question_type not in QUESTION_TYPE:
    raise ValueError("Invalid question type")
if args.model_id not in ["llama", "gemma", "claude", 'gpt']:
    raise ValueError("Invalid model id")
if args.experiment_type not in ["exp", "full", "add_prompt"]:
    raise ValueError("Invalid experiment type")


# # openAI initialization
openai.organization = "YOUR ORGANIZATION"
openai.api_version = "YOUR API VERSION"
openai.api_key = "YOUR API KEY"


STOP_AFTER_ATTEMPT=10

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))
def chat_gpt_call_vanilla(content, type):
    

    response = openai.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        
        messages = [
                {"role": "system", "content": "Please follow the instructions carefully."},
                {"role": "user", "content": content}
            ],
        temperature=.0000000000000000000001,
        top_p=.0000000000000000000001,   
        max_tokens=800,     
        frequency_penalty=0,
        presence_penalty=0,
        stop=["# END"] 
        )
    

    answer = response.choices[0].message.content.strip() 

    return answer

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_chain(*[wait_fixed(3) for i in range(2)] +
                       [wait_fixed(5) for i in range(1)]))

def evaluate_answers(input_file, output_file,input_type, continue_eval=False):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get previously processed characters if continuing evaluation
    processed_characters = set()
    if continue_eval and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                # Handle potential incomplete JSON by removing the trailing comma and bracket
                if content.endswith(',\n'):
                    content = content[:-2] + ']'
                elif not content.endswith(']'):
                    content = content + ']'
                processed_data = json.loads(content)
                processed_characters = {item["character"] for item in processed_data if "character" in item}
        except json.JSONDecodeError:
            # If the file is empty or has invalid JSON, create a new file
            with open(output_file, 'w') as f:
                f.write('[')

    # Start a new file if not continuing or if processed file doesn't exist
    if not continue_eval or not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write('[')

    for character_data in data:
        
        character = character_data.get("character")

        # Skip already processed characters if continuing
        if continue_eval and character in processed_characters:
            continue


        type = character_data.get("type")
        questions = character_data.get("questions", [])

        evaluations = []
        

        for question in tqdm(questions, desc=f"Evaluating {character}"):
            # Construct the prompt
            
            prompt = ACC_EVAL_PROMPT.format(question=question)
            answer = chat_gpt_call_vanilla(prompt,input_type)

            # Append the evaluation
            evaluations.append(answer)
        

        # Write the character's data to the output file after processing all questions
        with open(output_file, 'a') as f:  # Open in append mode
            json.dump({
                "character": character,
                "type": type,
                "questions": questions,
                f"{input_type} evaluation": evaluations
            }, f, indent=4)
            f.write(',')
            f.write('\n')  # Write a newline for each entry


    with open(output_file, 'r') as f:
        content = f.read()
    
    with open(output_file, 'a') as f:
        # If the file has only '[', don't add the closing bracket
        if len(content.strip()) > 1:  
            f.write(']')
        else:
            # If empty, write an empty array
            f.seek(0)
            f.write('[]')
            f.truncate()
     
cfg = {
    "type": args.type,
    "question_type": args.question_type
}

if __name__ == "__main__":
    type = cfg["type"]
    question_type = cfg["question_type"]
    dir_path = f"evaluation/{args.experiment_type}/{args.model_id}/{type}/{question_type}"
    os.makedirs(dir_path, exist_ok=True)

    input_type = "accuracy"

    evaluate_answers(f'answers/{args.experiment_type}/{args.model_id}/{type}/formatted_output_{type}_{question_type}.json', f'{dir_path}/evaluated_accuracy_{type}_{question_type}.json',input_type, args.continue_eval)


