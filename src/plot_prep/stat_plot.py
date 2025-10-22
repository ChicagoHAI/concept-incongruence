import json
import numpy as np
import os
import argparse
import sys
from scipy import stats  # Import for t-test
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.constant import TYPE, QUESTION_TYPE

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--question_type", type=str, default="four_president_given_current")
parser.add_argument("--model_id", type=str, default="llama")
parser.add_argument("--experiment_type", type=str, default="exp")
args = parser.parse_args()

if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.question_type not in QUESTION_TYPE:
    raise ValueError("Invalid question type")
if args.model_id not in ["llama", "gemma", "claude", "gpt"]:
    raise ValueError("Invalid model id")

def calculate_stats(file_path):
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize categories
    real_person_chars = []
    # mix_fictional_chars = []
    fully_fictional_chars = []
    
    # For t-test (only real person)
    real_person_conditional_accs = []
    
    # Group characters by category
    for char in data:
        if char["character_type"] == "real_person":
            real_person_chars.append(char)
        elif char["character_type"] == "fully_fictional":
            fully_fictional_chars.append(char)
    
    # Results dictionary
    results = {
        "real_person": {"count": len(real_person_chars)},
        # "mix_fictional": {"count": len(mix_fictional_chars)},
        "fully_fictional": {"count": len(fully_fictional_chars)}
    }
    
    # Calculate stats for fully fictional characters (no death consideration)
    if fully_fictional_chars and "yearly" not in args.question_type:
        abstain_rates = []
        answer_rates = []
        conditional_accs = []
        conditional_answers = []
        overall_accs = []
        
        for char in fully_fictional_chars:
            abstain_list = char["abstain_list"]
            answer_list = char["answer_list"]
            acc_list = char["acc_list"]
            
            abstain_rate = np.mean(abstain_list)
            answer_rate = np.mean(answer_list)
            
            # Calculate conditional accuracy (when abstain=0)
            cond_indices = [i for i, a in enumerate(abstain_list) if a == 0]
            
            if cond_indices:  # If there are any non-abstain indices
                conditional_acc = np.mean([acc_list[i] for i in cond_indices])
                conditional_accs.append(conditional_acc)
            
            # Calculate conditional answer (when abstain=1)
            abstain_indices = [i for i, a in enumerate(abstain_list) if a == 1]
            if abstain_indices:  # If there are any abstain indices
                conditional_answer = np.mean([answer_list[i] for i in abstain_indices])
                conditional_answers.append(conditional_answer)
            
            # Overall accuracy (all indices)
            overall_acc = np.mean(acc_list)
            
            abstain_rates.append(abstain_rate)
            answer_rates.append(answer_rate)
            overall_accs.append(overall_acc)
        
        results["fully_fictional"]["abstain"] = np.mean(abstain_rates)
        results["fully_fictional"]["answer"] = np.mean(answer_rates)
        results["fully_fictional"]["conditional_accuracy"] = np.mean(conditional_accs) if conditional_accs else None
        results["fully_fictional"]["conditional_answer"] = np.mean(conditional_answers) if conditional_answers else None
        results["fully_fictional"]["overall_accuracy"] = np.mean(overall_accs)
    
    for category, chars in [("real_person", real_person_chars)]:
        if not chars:
            continue
            
        # Initialize counters
        before_death_abstain = []
        before_death_answer = []
        before_death_cond_acc = []
        before_death_cond_answer = []
        before_death_overall_acc = []
        
        after_death_abstain = []
        after_death_answer = []
        after_death_cond_acc = []
        after_death_cond_answer = []
        
        # Add overall stats (regardless of death status)
        overall_abstain = []
        overall_answer = []
        overall_cond_acc = []
        overall_cond_answer = []
        overall_acc = []
        
        for char in chars:
            # Extract lists
            death_list = char["death_list"]
            abstain_list = char["abstain_list"]
            answer_list = char["answer_list"]
            acc_list = char["acc_list"]
            
            # Before death stats (death_list = 0)
            before_indices = [i for i, d in enumerate(death_list) if d == 0]
            if before_indices:  # If there are any indices before death
                before_death_abstain.append(np.mean([abstain_list[i] for i in before_indices]))
                before_death_answer.append(np.mean([answer_list[i] for i in before_indices]))
                
                # Calculate conditional accuracy (when abstain=0 and death=0)
                cond_indices = [i for i in before_indices if abstain_list[i] == 0]
                if cond_indices:  # If there are any non-abstain indices before death
                    before_death_cond_acc.append(np.mean([acc_list[i] for i in cond_indices]))
                
                # Calculate conditional answer (when abstain=1 and death=0)
                abstain_indices = [i for i in before_indices if abstain_list[i] == 1]
                if abstain_indices:  # If there are any abstain indices before death
                    before_death_cond_answer.append(np.mean([answer_list[i] for i in abstain_indices]))
                
                # Overall accuracy (when death=0)
                before_death_overall_acc.append(np.mean([acc_list[i] for i in before_indices]))
            
            # After death stats (death_list = 1)
            after_indices = [i for i, d in enumerate(death_list) if d == 1]
            if after_indices:  # If there are any indices after death
                after_death_abstain.append(np.mean([abstain_list[i] for i in after_indices]))
                after_death_answer.append(np.mean([answer_list[i] for i in after_indices]))

                # Calculate conditional accuracy (when abstain=0 and death=1)
                cond_indices = [i for i in after_indices if abstain_list[i] == 0]
                if cond_indices:  # If there are any non-abstain indices after death
                    after_death_cond_acc.append(np.mean([acc_list[i] for i in cond_indices]))
                
                # Calculate conditional answer (when abstain=1 and death=1)
                abstain_indices = [i for i in after_indices if abstain_list[i] == 1]
                if abstain_indices:  # If there are any abstain indices after death
                    after_death_cond_answer.append(np.mean([answer_list[i] for i in abstain_indices]))
            
            # Overall stats (regardless of death status)
            overall_abstain.append(np.mean(abstain_list))
            overall_answer.append(np.mean(answer_list))
            
            # Calculate overall conditional accuracy (when abstain=0)
            cond_indices = [i for i, a in enumerate(abstain_list) if a == 0]
            if cond_indices:
                char_cond_accs = [acc_list[i] for i in cond_indices]
                overall_cond_acc.append(np.mean(char_cond_accs))
                
                # Only collect real person conditional accuracies for t-test
                if category == "real_person":
                    real_person_conditional_accs.extend(char_cond_accs)
            
            # Calculate overall conditional answer (when abstain=1)
            abstain_indices = [i for i, a in enumerate(abstain_list) if a == 1]
            if abstain_indices:
                overall_cond_answer.append(np.mean([answer_list[i] for i in abstain_indices]))
            
            # Overall accuracy
            overall_acc.append(np.mean(acc_list))
        
        # Store results
        results[category]["before_death"] = {
            "abstain": np.mean(before_death_abstain) if before_death_abstain else None,
            "answer": np.mean(before_death_answer) if before_death_answer else None,
            "conditional_accuracy": np.mean(before_death_cond_acc) if before_death_cond_acc else None,
            "conditional_answer": np.mean(before_death_cond_answer) if before_death_cond_answer else None,
            "overall_accuracy": np.mean(before_death_overall_acc) if before_death_overall_acc else None
        }
        
        results[category]["after_death"] = {
            "abstain": np.mean(after_death_abstain) if after_death_abstain else None,
            "answer": np.mean(after_death_answer) if after_death_answer else None,
            "conditional_accuracy": np.mean(after_death_cond_acc) if after_death_cond_acc else None,
            "conditional_answer": np.mean(after_death_cond_answer) if after_death_cond_answer else None
        }
        
        # Add overall results (regardless of death status)
        results[category]["overall"] = {
            "abstain": np.mean(overall_abstain) if overall_abstain else None,
            "answer": np.mean(overall_answer) if overall_answer else None,
            "conditional_accuracy": np.mean(overall_cond_acc) if overall_cond_acc else None,
            "conditional_answer": np.mean(overall_cond_answer) if overall_cond_answer else None,
            "overall_accuracy": np.mean(overall_acc) if overall_acc else None
        }
    
    # Add t-test results to the statistics (only real person)
    results["t_test"] = {
        "real_person_conditional_accs": real_person_conditional_accs,
        "count": len(real_person_conditional_accs)
    }
    
    return results

def print_results(results):
    print("=== Statistical Analysis ===")
    
    # Print fully fictional results if not yearly question type
    if "yearly" not in args.question_type:
        print("\nFully Fictional Characters (Count: {})".format(results["fully_fictional"]["count"]))
        print("  Abstain Rate: {:.3f}".format(results["fully_fictional"]["abstain"] if "abstain" in results["fully_fictional"] else 0))
        print("  Answer Rate: {:.3f}".format(results["fully_fictional"]["answer"] if "answer" in results["fully_fictional"] else 0))
        print("  Conditional Accuracy: {:.3f}".format(
            results["fully_fictional"]["conditional_accuracy"] if "conditional_accuracy" in results["fully_fictional"] and results["fully_fictional"]["conditional_accuracy"] is not None else 0))
        print("  Conditional Answer: {:.3f}".format(
            results["fully_fictional"]["conditional_answer"] if "conditional_answer" in results["fully_fictional"] and results["fully_fictional"]["conditional_answer"] is not None else 0))
        print("  Overall Accuracy: {:.3f}".format(results["fully_fictional"]["overall_accuracy"] if "overall_accuracy" in results["fully_fictional"] else 0))
    
    # Print real person results
    print("\nReal Person Characters (Count: {})".format(results["real_person"]["count"]))
    
    # Add overall stats section for real person
    print("  Overall (Regardless of Death Status):")
    overall = results["real_person"]["overall"]
    print("    Abstain Rate: {:.3f}".format(overall["abstain"] if overall["abstain"] is not None else 0))
    print("    Answer Rate: {:.3f}".format(overall["answer"] if overall["answer"] is not None else 0))
    print("    Conditional Accuracy: {:.3f}".format(overall["conditional_accuracy"] if overall["conditional_accuracy"] is not None else 0))
    print("    Conditional Answer: {:.3f}".format(overall["conditional_answer"] if overall["conditional_answer"] is not None else 0))
    print("    Overall Accuracy: {:.3f}".format(overall["overall_accuracy"] if overall["overall_accuracy"] is not None else 0))
    
    print("  Before Death:")
    bd = results["real_person"]["before_death"]
    print("    Abstain Rate: {:.3f}".format(bd["abstain"] if bd["abstain"] is not None else 0))
    print("    Answer Rate: {:.3f}".format(bd["answer"] if bd["answer"] is not None else 0))
    print("    Conditional Accuracy: {:.3f}".format(bd["conditional_accuracy"] if bd["conditional_accuracy"] is not None else 0))
    print("    Conditional Answer: {:.3f}".format(bd["conditional_answer"] if bd["conditional_answer"] is not None else 0))
    print("    Overall Accuracy: {:.3f}".format(bd["overall_accuracy"] if bd["overall_accuracy"] is not None else 0))
    
    print("  After Death:")
    ad = results["real_person"]["after_death"]
    print("    Abstain Rate: {:.3f}".format(ad["abstain"] if ad["abstain"] is not None else 0))
    print("    Answer Rate: {:.3f}".format(ad["answer"] if ad["answer"] is not None else 0))
    print("    Conditional Accuracy: {:.3f}".format(ad["conditional_accuracy"] if ad["conditional_accuracy"] is not None else 0))
    print("    Conditional Answer: {:.3f}".format(ad["conditional_answer"] if ad["conditional_answer"] is not None else 0))
    
    # Print aggregated results of real person and mix fictional
    real_count = results["real_person"]["count"]
    # mix_count = results["mix_fictional"]["count"] 
    mix_count = 0  # Set to 0 since we're commenting out mix fictional
    total_count = real_count + mix_count
    
    # print("\nAggregated Real Person + Mix Fictional (Count: {})".format(total_count))
    print("\nReal Person Only (Count: {})".format(total_count))
    
    # Calculate weighted averages for before death
    if total_count > 0:
        print("  Before Death:")
        real_bd = results["real_person"]["before_death"]
        # mix_bd = results["mix_fictional"]["before_death"]
        
        # Helper function for weighted average calculation
        def weighted_avg(real_val, real_count):
            real_val = real_val if real_val is not None else 0
            if real_count > 0:
                return real_val

        # Use real person values directly
        abstain_bd = real_bd["abstain"] if real_bd["abstain"] is not None else 0
        answer_bd = real_bd["answer"] if real_bd["answer"] is not None else 0
        cond_acc_bd = real_bd["conditional_accuracy"] if real_bd["conditional_accuracy"] is not None else 0
        cond_answer_bd = real_bd["conditional_answer"] if real_bd["conditional_answer"] is not None else 0
        overall_acc_bd = real_bd["overall_accuracy"] if real_bd["overall_accuracy"] is not None else 0
        
        print("    Abstain Rate: {:.3f}".format(abstain_bd))
        print("    Answer Rate: {:.3f}".format(answer_bd))
        print("    Conditional Accuracy: {:.3f}".format(cond_acc_bd))
        print("    Conditional Answer: {:.3f}".format(cond_answer_bd))
        print("    Overall Accuracy: {:.3f}".format(overall_acc_bd))
        
        
        # Calculate weighted averages for after death
        print("  After Death:")
        real_ad = results["real_person"]["after_death"]
        # Use real person values directly
        abstain_ad = real_ad["abstain"] if real_ad["abstain"] is not None else 0
        answer_ad = real_ad["answer"] if real_ad["answer"] is not None else 0
        cond_acc_ad = real_ad["conditional_accuracy"] if real_ad["conditional_accuracy"] is not None else 0
        cond_answer_ad = real_ad["conditional_answer"] if real_ad["conditional_answer"] is not None else 0
        
        print("    Abstain Rate: {:.3f}".format(abstain_ad))
        print("    Answer Rate: {:.3f}".format(answer_ad))
        print("    Conditional Accuracy: {:.3f}".format(cond_acc_ad))
        print("    Conditional Answer: {:.3f}".format(cond_answer_ad))

    # Add overall aggregated stats
    # print("\nAggregated Overall (Real Person + Mix Fictional):")
    print("\nReal Person Overall:")
    real_overall = results["real_person"]["overall"]
    # mix_overall = results["mix_fictional"]["overall"]
    
    # Helper function for weighted average calculation
    def weighted_avg(real_val, real_count):
        real_val = real_val if real_val is not None else 0
        if real_count > 0:
            return real_val
        return 0

    # Use real person values directly
    abstain_overall = real_overall["abstain"] if real_overall["abstain"] is not None else 0
    answer_overall = real_overall["answer"] if real_overall["answer"] is not None else 0
    cond_acc_overall = real_overall["conditional_accuracy"] if real_overall["conditional_accuracy"] is not None else 0
    cond_answer_overall = real_overall["conditional_answer"] if real_overall["conditional_answer"] is not None else 0
    acc_overall = real_overall["overall_accuracy"] if real_overall["overall_accuracy"] is not None else 0
    
    print("  Abstain Rate: {:.3f}".format(abstain_overall))
    print("  Answer Rate: {:.3f}".format(answer_overall))
    print("  Conditional Accuracy: {:.3f}".format(cond_acc_overall))
    print("  Conditional Answer: {:.3f}".format(cond_answer_overall))
    print("  Overall Accuracy: {:.3f}".format(acc_overall))
    
    # Add t-test comparison against perfect accuracy (only for real person)
    if "t_test" in results and results["t_test"]["real_person_conditional_accs"]:
        real_person_accs = results["t_test"]["real_person_conditional_accs"]
        count = results["t_test"]["count"]
        perfect_accs = [1] * 27
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(real_person_accs, perfect_accs)
        
        print("\nT-test comparison with perfect accuracy (Real Person only):")
        print(f"  Number of samples: {count}")
        print(f"  Average conditional accuracy: {np.mean(real_person_accs):.3f}")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_value:.6f}")
        
        # Interpret the results
        alpha = 0.05
        print(f"  Significant difference from perfect accuracy (α={alpha}): {'Yes' if p_value < alpha else 'No'}")


cfg = {
    "type": args.type,
    "question_type": args.question_type,
}
if __name__ == "__main__":
    file_path = f"evaluation/{args.experiment_type}/{args.model_id}/{cfg['type']}/{cfg['question_type']}/final_evaluation_{cfg['type']}_{cfg['question_type']}.json"
    results = calculate_stats(file_path)
    print_results(results)