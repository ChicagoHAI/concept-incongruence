import json
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--question_type", type=str, default="commonsense")
parser.add_argument("--model_id", type=str, default="gpt")
parser.add_argument("--experiment_type", type=str, default="ai")
parser.add_argument("--continue_eval", action="store_true", help="Continue evaluation from previous run")
args = parser.parse_args()

# Load model predictions
if  "ai" in args.experiment_type:
    with open(f"answers/{args.experiment_type}/{args.model_id}/formatted_output_QA_{args.question_type}.json", "r") as f:
        model_data = json.load(f)
else:
    with open(f"answers/{args.experiment_type}/{args.model_id}/{args.type}/formatted_output_{args.type}_{args.question_type}.json", "r") as f:
        model_data = json.load(f)

# Load ground truth labels
with open(f"datasets/{args.question_type}_cot_partial_raw_new.json", "r") as f:
    ground_truth = json.load(f)

def extract_answer(question_text):
    """Extract the first capitalized character(s) after 'Answer:'"""
    # Find the part after "Answer:"
    answer_match = re.search(r'Answer:\s*([A-Z]+)', question_text)
    if answer_match:
        return answer_match.group(1)
    return None

# Extract ground truth answers
gt_answers = [item["answer"] for item in ground_truth]

# Calculate accuracy for each character
character_results = {}
gt_index = 0

print("=" * 60)
print("ACCURACY EVALUATION RESULTS")
print("=" * 60)

for item in model_data:
    character = item.get("character", "Unknown")
    questions = item.get("questions", [])
    
    
    # Extract model answers for this character
    model_answers = []
    for question in questions:
        answer = extract_answer(question)
        model_answers.append(answer)
    
    # Get corresponding ground truth answers
    num_questions = len(model_answers)
    gt_index += num_questions
    
    # Calculate accuracy for this character
    correct = 0
    total = 30
    
    print(f"\nCharacter: {character}")
    print("-" * 40)
    
    for i in range(total):
        model_ans = model_answers[i]
        gt_ans = gt_answers[i]
        is_correct = model_ans == gt_ans
        
        if is_correct:
            correct += 1
        
        print(f"Q{i+1}: Model={model_ans}, GT={gt_ans}, Correct={is_correct}")
    
    accuracy = correct / total if total > 0 else 0
    character_results[character] = {
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }
    
    print(f"Character {character} Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculate overall statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

total_correct = 0
total_questions = 0
character_accuracies = []

for character, results in character_results.items():
    total_correct += results['correct']
    total_questions += results['total']
    character_accuracies.append(results['accuracy'])
    print(f"{character}: {results['correct']}/{results['total']} = {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
average_character_accuracy = sum(character_accuracies) / len(character_accuracies) if character_accuracies else 0

print("-" * 40)
print(f"Overall Accuracy: {total_correct}/{total_questions} = {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Average Character Accuracy: {average_character_accuracy:.4f} ({average_character_accuracy*100:.2f}%)")
print(f"Number of Characters: {len(character_results)}")



