"""
Extract all the accuracy, abstain, and answer labels
Output into a json file
"""
import json
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.constant import TYPE, QUESTION_TYPE
from src.analysis.human_label_match import extract_labels_from_json
# from src.eval.evaluate import evaluate_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--question_type", type=str, default="four_president_given_death_current")
parser.add_argument("--model_id", type=str, default="llama")
parser.add_argument("--experiment_type", help="type of experiment (exp or full)", type=str, default="exp")
args = parser.parse_args()

if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.question_type not in QUESTION_TYPE:
    raise ValueError("Invalid question type")
if args.model_id not in ["llama", "gemma", "claude","gpt"]:
    raise ValueError("Invalid model id")
if args.experiment_type not in ["exp", "full", "add_prompt"]:
    raise ValueError("Invalid experiment type")


def extract_president_evaluation(save_path):
    # Read the question.json file
    if args.experiment_type == "exp":
        with open('answers/question.json', 'r') as f:
            data = json.load(f)
    elif args.experiment_type == "full" or args.experiment_type == "add_prompt":
        with open('answers/full_question.json', 'r') as f:
            data = json.load(f)
    
    # Dictionary mapping presidents to their terms
    presidents = {
        "Biden": (2021, 2025),
        "Trump": (2017, 2021),
        "Obama": (2009, 2017),
        "G.W. Bush": (2001, 2009),
        "Clinton": (1993, 2001),
        "G.H.W. Bush": (1989, 1993),
        "Reagan": (1981, 1989),
        "Carter": (1977, 1981),
        "Ford": (1974, 1977),
        "Nixon": (1969, 1974),
        "L.B. Johnson": (1963, 1969),
        "Kennedy": (1961, 1963),
        "Eisenhower": (1953, 1961),
        "Truman": (1945, 1953),
        "F.D. Roosevelt": (1933, 1945),
        "Hoover": (1929, 1933),
        "Coolidge": (1923, 1929),
        "Harding": (1921, 1923),
        "Wilson": (1913, 1921),
        "Taft": (1909, 1913),
        "T. Roosevelt": (1901, 1909),
        "McKinley": (1897, 1901),
        "Cleveland (2nd)": (1893, 1897),
        "B. Harrison": (1889, 1893),
        "Cleveland": (1885, 1889),
        "Arthur": (1881, 1885),
        "Garfield": (1881, 1881)
    }
    
    # Process each category in the data
    output = []
    for category in data:
        characters = category.get("characters", [])
        
        # Process each character in the category
        for character in characters:
            death_year = character.get("year")
            
            # If no death year is provided, set all presidents as serving after death (all 1s)
            if not death_year:
                president_evaluation = [1] * len(presidents)
                output.append({
                    "character": character.get("role"),
                    "death_list": president_evaluation,
                    "character_type": character.get("type"),
                    "abstain_list": [],
                    "answer_list": [],
                    "acc_list": []
                })
                continue
            
            # Convert death year to integer if it's a string
            if isinstance(death_year, str):
                try:
                    death_year = int(death_year)
                except ValueError:
                    # If death year can't be converted to int, set all presidents as after death
                    president_evaluation = [1] * len(presidents)
                    output.append({
                        "character": character.get("role"),
                        "death_list": president_evaluation,
                        "character_type": character.get("type"),
                        "abstain_list": [],
                        "answer_list": [],
                        "acc_list": []
                    })
                    continue
            
            # Determine which presidents served before/during (0) or after (1) the character's death
            president_evaluation = []
            
            for president, term in presidents.items():
                start_year, end_year = term
                
                # If the president's term ended before or during the death year, mark as 0
                # Otherwise, mark as 1
                if end_year <= death_year:
                    president_evaluation.append(0)  # President served before or during death year
                else:
                    president_evaluation.append(1)  # President served after death year
            
            # Add the evaluation to the character data
            output.append({
                "character": character.get("role"),
                "death_list": president_evaluation,
                "character_type": character.get("type"),
                "abstain_list": [],
                "answer_list": [],
                "acc_list": []
            })
    
    # Write the updated data back to the file
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)

def combine_evaluation(acc_path, abstain_list, answer_list,save_path, intermediate_evaluation_path):
    # Load accuracy data
    with open(acc_path, 'r') as f:
        acc_data = json.load(f)
    
    # Load death list data from the president evaluation
    with open(intermediate_evaluation_path, 'r') as f:
        death_data = json.load(f)

    # Create a mapping of character names to death lists
    character_death_lists = {}
    character_type_lists = {}
    for item in death_data:
        character = item.get("character")
        death_list = item.get("death_list")
        character_type = item.get("character_type")
        if character and death_list:
            character_death_lists[character] = death_list
            character_type_lists[character] = character_type
    
    # Read the question data to get character death years
    if args.experiment_type == "exp":
        with open('answers/question.json', 'r') as f:
            question_data = json.load(f)
    elif args.experiment_type == "full" or args.experiment_type == "add_prompt":
        with open('answers/full_question.json', 'r') as f:
            question_data = json.load(f)
    
    # Create a mapping of character names to death years
    character_death_years = {}
    for category in question_data:
        for character in category.get("characters", []):
            role = character.get("role")
            death_year = character.get("year")
            if role and death_year:
                try:
                    if isinstance(death_year, str):
                        death_year = int(death_year)
                    character_death_years[role] = death_year
                except ValueError:
                    continue
    
    # Combine all data
    output = []
    cond_avg_list = []
    for i, acc_item in enumerate(acc_data):
        persona = acc_item.get("character")
        
        # Skip if we don't have matching data
        if i >= len(abstain_list) or i >= len(answer_list):
            continue
        
        # Get the corresponding abstain and answer lists
        abstain = abstain_list[persona]
        answers = answer_list[persona]
        
        # Get death year for this character
        death_year = character_death_years.get(persona)
        
        # Get death list for this character
        death_list = character_death_lists.get(persona, [])
        if cfg["question_type"] == "yearly" or cfg["question_type"] == "yearly_given_death":
            death_list = [0] * 31 + [1] * 30
            
        character_type = character_type_lists.get(persona, "")
        # breakpoint()
        # Create combined entry
        acc_label = []
        for acc_answer in acc_item["accuracy evaluation"]:
            acc_label.append(int(acc_answer[0]))
        combined_item = {
            "character": persona,
            "character_type": character_type,
            "death_year": death_year,
            "death_list": death_list,
            "abstain_list": abstain,
            "answer_list": answers,
            "acc_list": acc_label
        }
        print("Name: ", persona)
        sum_acc = 0
        count = 0   
        # breakpoint()
        #FIXME
        # death_list = [0] * 27
        for i in range(len(acc_item["accuracy evaluation"])):
            if death_list[i] == 0:
                sum_acc += acc_label[i]
                count += 1
        if count > 0:
            print("Sum of accuracy: ", sum_acc/count)
        else:
            print("Sum of accuracy: ", 0)
            
        # Compare acc_label with [1]*27 and calculate p-value
        from scipy import stats
        import numpy as np
        
        # Ensure acc_label is the right length for comparison
        acc_array = np.array(acc_label[:27]) if len(acc_label) >= 27 else np.pad(acc_label, (0, 27-len(acc_label)))
        ones_array = np.ones(27)
        
        # Calculate difference
        diff = acc_array - ones_array
        
        # Perform t-test to get p-value
        t_stat, p_value = stats.ttest_1samp(diff, 0)
        print(f"Comparison with [1]*27: p-value = {p_value:.6f}")
        
        conditional_acc = []
        for i, abs_answer in enumerate(abstain):
            if abs_answer == 0:
                conditional_acc.append(acc_label[i])
        if len(conditional_acc) == 0:
            cond_avg = 0
        else:
            cond_avg = sum(conditional_acc)/len(conditional_acc)
        print("Conditional accuracy: ", cond_avg)
        if len(conditional_acc) > 0:
            cond_avg_list.append(cond_avg)
        output.append(combined_item)
    
    print("Conditional accuracy: ", sum(cond_avg_list)/len(cond_avg_list))
    
    # Compare overall conditional accuracy with [1]*27
    all_conditional_acc = []
    for item in output:
        for i, abs_val in enumerate(item["abstain_list"]):
            if abs_val == 0 and i < len(item["acc_list"]):
                all_conditional_acc.append(item["acc_list"][i])
    
    if all_conditional_acc:
        # Perform t-test comparing all conditional accuracy values with constant 1
        all_acc_array = np.array(all_conditional_acc)
        ones_array = np.ones(len(all_conditional_acc))
        diff_all = all_acc_array - ones_array
        t_stat_all, p_value_all = stats.ttest_1samp(diff_all, 0)
        print(f"Overall conditional accuracy: {np.mean(all_acc_array):.4f}")
        print(f"Overall comparison with [1]*27: p-value = {p_value_all:.6f}")
    
    # Write the combined data to a file
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)

cfg = {
    "type": args.type,
    "question_type": args.question_type
}

if __name__ == "__main__":
    type = cfg["type"]
    question_type = cfg["question_type"]
    dir_path = f"evaluation/{args.experiment_type}/{args.model_id}/{type}/{question_type}"
    os.makedirs(dir_path, exist_ok=True)


    # intermediate_evaluation_path = f"{dir_path}/intermediate_evaluation.json"
    if args.experiment_type == "exp":
        intermediate_evaluation_path = "evaluation/intermediate_evaluation.json" 
    elif args.experiment_type == "full" or args.experiment_type == "add_prompt":
        intermediate_evaluation_path = "evaluation/full_death_list.json"
    answer_path = f"answers/{args.experiment_type}/{args.model_id}/{type}/formatted_output_{type}_{question_type}.json"
    accuracy_path = f"{dir_path}/evaluated_accuracy_{type}_{question_type}.json"
    llm_evaluation_path = f"{dir_path}/{type}_abstain_answer.json"
    final_evaluation_path = f"{dir_path}/final_evaluation_{type}_{question_type}.json"

    ## evaluate accuracy
    # extract_president_evaluation(intermediate_evaluation_path)
    # evaluate_accuracy(answer_path, accuracy_path)
    ## extract abstain and answer labels
    abstract_labels, answer_labels = extract_labels_from_json(llm_evaluation_path)
    ## combine evaluation
    combine_evaluation(accuracy_path, abstract_labels, answer_labels, final_evaluation_path, intermediate_evaluation_path)