"""
Evaluate for the yearly question,
no need to run the extract_evaluation.py
"""
import json
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
# sys.path.append("Persona_Understanding")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.constant import TYPE, QUESTION_TYPE

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--question_type", type=str, default="four_president_given_death_current")
parser.add_argument("--model_id", type=str, default="llama")
parser.add_argument("--plot_type", type=str, default="results", choices=["results", "ideal", "both"], 
                    help="Choose to plot 'results', 'ideal', or 'both' separately")
parser.add_argument("--experiment_type", type=str, default="full", choices=["full", "exp"], 
                    help="Choose to plot 'full', or 'die'")
args = parser.parse_args()

sns.set_theme(style="white")
sns.set_context("talk", font_scale=6)

if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.question_type not in ["yearly","yearly_cot","yearly_given_death"]:
    raise ValueError("Invalid question type")
if args.model_id not in ["llama", "gemma", "claude", 'gpt']:
    raise ValueError("Invalid model id")

cfg = {
    "type": args.type,
    "question_type": args.question_type
}

abstain_ideal = [0] * 31 + [1] * 31
answer_ideal = [1] * 31 + [0] * 31
acc_ideal = [1]*61

def extract_metrics(data):
    acc_list = []
    abstain_list = []
    answer_list = []
    for char_data in data:
        acc_list.append(char_data["acc_list"])
        abstain_list.append(char_data["abstain_list"])
        answer_list.append(char_data["answer_list"])
    return torch.tensor(acc_list,dtype=torch.float32), torch.tensor(abstain_list,dtype=torch.float32), torch.tensor(answer_list,dtype=torch.float32)

def calculate_conditional_accuracy(acc_list, abstain_list):
    avg_acc = []
    for i in range(acc_list.shape[1]):
        avg_acc.append(acc_list[:,i][abstain_list[:,i] == 0].mean(0))
        # breakpoint()
    # breakpoint()
    return torch.tensor(avg_acc, dtype=torch.float32)

if __name__ == "__main__":
    # Set Seaborn theme and context
    sns.set_theme(style="white")
    sns.set_context("talk", font_scale=6)

    type = cfg["type"]
    question_type = cfg["question_type"]

    # Define color scheme for consistency between plots
    metric_colors = {'abstain_rate': '#1f77b4', 'answer_rate': '#ff7f0e', 'accuracy': '#2ca02c'}
    
    # Define the x-ticks and labels
    x_ticks = [0, 10, 20, 30, 40, 50, 60]  # Assuming 61 data points
    x_labels = ['-30', '-20', '-10', '0', '+10', '+20', '+30']

    if args.plot_type in ["results", "both"]:
        with open(os.path.join(f"evaluation/{args.experiment_type}/{args.model_id}/{type}/{question_type}/final_evaluation_{type}_{question_type}.json"), "r") as f:
            data = json.load(f)
        
        acc_list, abstain_list, answer_list = extract_metrics(data)
        avg_acc = acc_list.mean(0)
        avg_abstain = abstain_list.mean(0)
        avg_answer = answer_list.mean(0)

        # Calculate conditional accuracy
        conditional_avg_acc = calculate_conditional_accuracy(acc_list, abstain_list)

        # Set the figure size for results plot
        plt.figure(figsize=(24,20))

        # Plotting the results with Seaborn
        # sns.lineplot(data=conditional_avg_acc.numpy()[::-1], label='Conditional Accuracy', marker='x', color=metric_colors['accuracy'], linewidth=5, markersize=10)
        sns.lineplot(data=avg_abstain.numpy()[::-1], label='Abstain', marker='s', color=metric_colors['abstain_rate'], linewidth=5, markersize=10)
        sns.lineplot(data=avg_answer.numpy()[::-1], label='Answer', marker='^', color=metric_colors['answer_rate'], linewidth=5, markersize=10)

        # Mark the middle position (31st)
        plt.axvline(x=30, color='r', linestyle='--', label='Death Year', linewidth=4)
        
        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel('Index')
        plt.ylabel('Rate (%)')
        
        # Format y-axis as percentages first
        yticks = plt.gca().get_yticks()
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels([f'{int(x*100)}' for x in yticks])
        
        # THEN set y-axis limits from 0 to 105% - after formatting ticks
        plt.ylim(0, 1.05)
        
        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.grid(False)
        plt.savefig(f'plot/{args.experiment_type}/{args.model_id}_{type}_{question_type}_results.pdf', bbox_inches='tight', format='pdf')
        plt.close()

    if args.plot_type in ["ideal", "both"]:
        # Set the figure size for ideal plot
        plt.figure(figsize=(24,20))
        
        # Plotting ideal cases
        # sns.lineplot(data=acc_ideal, label='Conditional Accuracy',  color=metric_colors['accuracy'], linewidth=5 )
        sns.lineplot(data=abstain_ideal, label='Abstain', color=metric_colors['abstain_rate'], linewidth=5)
        sns.lineplot(data=answer_ideal, label='Answer', color=metric_colors['answer_rate'], linewidth=5)

        # Mark the middle position (31st)
        plt.axvline(x=30, color='r', linestyle='--', label='Death Year', linewidth=5)
        
        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel('Index')
        plt.ylabel('Rate (%)')
        
        # Format y-axis as percentages first
        yticks = plt.gca().get_yticks()
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels([f'{int(x*100)}' for x in yticks])
        
        # THEN set y-axis limits from 0 to 105% - after formatting ticks
        plt.ylim(0, 1.05)
        
        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.grid(False)
        plt.savefig(f'plot/{args.experiment_type}/{args.model_id}_{type}_{question_type}_ideal.pdf', bbox_inches='tight', format='pdf')
        plt.close()

