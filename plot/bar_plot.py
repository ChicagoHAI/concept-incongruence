import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Persona_Understanding.src.constant import TYPE, QUESTION_TYPE

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="normal")
parser.add_argument("--question_type", type=str, default="four_president_given_death_current")
args = parser.parse_args()

if args.type not in TYPE:
    raise ValueError("Invalid type")
if args.question_type not in QUESTION_TYPE:
    raise ValueError("Invalid question type")

def read_stat_file(file_path):
    """Read and parse the stat file to extract metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract data for each character type
    data = {}
    
    # Extract fully fictional data
    fully_fictional_match = re.search(r'Fully Fictional Characters.*?Abstain Rate: ([\d\.]+).*?Answer Rate: ([\d\.]+).*?Conditional Accuracy: ([\d\.]+)', 
                                      content, re.DOTALL)
    if fully_fictional_match:
        data['Fully Fictional'] = {
            'abstain_rate': float(fully_fictional_match.group(1)),
            'answer_rate': float(fully_fictional_match.group(2)),
            'accuracy': float(fully_fictional_match.group(3)),
            'baseline_abstain_rate': None,
            'baseline_answer_rate': None
        }
    
    # Extract real person data
    real_person_match = re.search(r'Real Person Characters.*?Before Death:.*?Abstain Rate: ([\d\.]+).*?Answer Rate: ([\d\.]+).*?Conditional Accuracy: ([\d\.]+).*?After Death:.*?Abstain Rate: ([\d\.]+).*?Answer Rate: ([\d\.]+)', 
                                 content, re.DOTALL)
    if real_person_match:
        data['Real Person'] = {
            'baseline_abstain_rate': float(real_person_match.group(1)),
            'baseline_answer_rate': float(real_person_match.group(2)),
            'accuracy': float(real_person_match.group(3)),
            'abstain_rate': float(real_person_match.group(4)),
            'answer_rate': float(real_person_match.group(5))
        }
    
    # Extract mix fictional data
    mix_fictional_match = re.search(r'Mix Fictional Characters.*?Before Death:.*?Abstain Rate: ([\d\.]+).*?Answer Rate: ([\d\.]+).*?Conditional Accuracy: ([\d\.]+).*?After Death:.*?Abstain Rate: ([\d\.]+).*?Answer Rate: ([\d\.]+)', 
                                   content, re.DOTALL)
    if mix_fictional_match:
        data['Mix Fictional'] = {
            'baseline_abstain_rate': float(mix_fictional_match.group(1)),
            'baseline_answer_rate': float(mix_fictional_match.group(2)),
            'accuracy': float(mix_fictional_match.group(3)),
            'abstain_rate': float(mix_fictional_match.group(4)),
            'answer_rate': float(mix_fictional_match.group(5))
        }
    
    return data

def plot_character_metrics(data, type, output_path=None, question=None):
    """Create a bar plot of character metrics."""
    # Ensure specific order: Real Person, Mix Fictional, Fully Fictional
    ordered_types = []
    if "Real Person" in data:
        ordered_types.append("Real Person")
    if "Mix Fictional" in data:
        ordered_types.append("Mix Fictional")
    # Only include Fully Fictional if not a given_death question
    if "Fully Fictional" in data and (question is None or "given_death" not in question):
        ordered_types.append("Fully Fictional")
    
    character_types = ordered_types
    
    # Extract data for plotting
    abstain_rates = [data[char_type]['abstain_rate'] for char_type in character_types]
    answer_rates = [data[char_type]['answer_rate'] for char_type in character_types]
    accuracies = [data[char_type]['accuracy'] for char_type in character_types]
    
    # Baseline data
    baseline_abstain_rates = [data[char_type].get('baseline_abstain_rate') for char_type in character_types]
    baseline_answer_rates = [data[char_type].get('baseline_answer_rate') for char_type in character_types]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set width of bars
    bar_width = 0.2
    index = np.arange(len(character_types))
    
    # Use colors from stat.py
    metric_colors = {
        'abstain_rate': '#3d88ad',  # Blue
        'answer_rate': '#f2b56e',   # Orange/yellow
        'accuracy': '#8ab07c'       # Purple/pink (using conditional_accuracy color)
    }
    
    baseline_colors = {
        'abstain_rate': '#1f77b4',  # Different blue for baseline
        'answer_rate': '#ff7f0e'    # Different orange for baseline
    }
    
    # Create bars
    bars1 = ax.bar(index - bar_width, abstain_rates, bar_width, color=metric_colors['abstain_rate'], label='Average abstain_rate')
    bars2 = ax.bar(index, answer_rates, bar_width, color=metric_colors['answer_rate'], label='Average answer_rate')
    bars3 = ax.bar(index + bar_width, accuracies, bar_width, color=metric_colors['accuracy'], label='Average accuracy')
    
    # Create baseline lines for abstain and answer rates
    for i, (char_type, abs_rate, ans_rate) in enumerate(zip(character_types, baseline_abstain_rates, baseline_answer_rates)):
        if abs_rate is not None:
            ax.plot([i - bar_width - 0.1, i + bar_width + 0.1], [abs_rate, abs_rate], 
                    color=baseline_colors['abstain_rate'], linestyle='--', linewidth=4, 
                    label='Baseline abstain_rate' if i == 0 else "")
        if ans_rate is not None:
            ax.plot([i - bar_width - 0.1, i + bar_width + 0.1], [ans_rate, ans_rate], 
                    color=baseline_colors['answer_rate'], linestyle='--', linewidth=4,
                    label='Baseline answer_rate' if i == 0 else "")
    
    # Add labels, title and custom x-axis tick labels with consistent font sizes
    ax.set_xlabel('Character Type', fontsize=26)
    ax.set_ylabel('Rate', fontsize=26)
    if type == "die":
        ax.set_title('Character Behavioral Metrics with Restricted Prompts', fontsize=28)
    else:
        ax.set_title('Character Behavioral Metrics with Non-Restricted Prompts', fontsize=28)
    ax.set_xticks(index)
    ax.set_xticklabels(character_types, fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 1.05)  # Set y-axis limit
    ax.legend(fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

cfg = {
    "type": args.type,
    "question": args.question_type,
}
if __name__ == "__main__":
    # Path to the stat file
    stat_file = f"evaluation/{cfg['type']}/{cfg['question']}/stat_{cfg['type']}_{cfg['question']}.txt"
    
    # Read data
    data = read_stat_file(stat_file)
    
    # Create output directory if it doesn't exist
    output_dir = f"plot/outputs/{cfg['type']}/{cfg['question']}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save
    output_path = os.path.join(output_dir, "character_behavioral_metrics.png")
    plot_character_metrics(data, cfg['type'], output_path, cfg['question'])
    
    print(f"Plot saved to {output_path}")
