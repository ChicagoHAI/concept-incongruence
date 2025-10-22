import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse
import seaborn as sns

parser = argparse.ArgumentParser(description='Plot abstain rates from statistics file.')
parser.add_argument('--question_type', type=str, default='four_president', help='Question type')
parser.add_argument('--model_id', type=str, default='llama', help='Model id')
parser.add_argument('--type', type=str, default='normal', help='Type')
parser.add_argument('--fictional_type', choices=['real', 'mix', 'fully', 'aggregated'], default='real',
                    help='Character type to plot (real, mix, fully, or aggregated)')
parser.add_argument('--metric', type=str, choices=['abstain', 'answer', 'accuracy'], 
                    default='abstain', help='Metric to plot (abstain, answer, or conditional accuracy)')
parser.add_argument('--experiment_type', type=str, default='full', help='Experiment type')

args = parser.parse_args()

# Use seaborn style
sns.set_theme(style="white")
sns.set_context("talk", font_scale=2.5)

def parse_stat_file(filepath):
    """Parse the statistics file and extract all metrics."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    stats = {}
    
    # Extract fully fictional metrics
    fully_fictional_match = re.search(r"Fully Fictional Characters.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    if fully_fictional_match:
        stats['fully_fictional_abstain'] = float(fully_fictional_match.group(1))
        stats['fully_fictional_answer'] = float(fully_fictional_match.group(2))
        stats['fully_fictional_accuracy'] = float(fully_fictional_match.group(3))
    
    # Extract real person metrics
    real_before_match = re.search(r"Real Person Characters.*?Before Death:.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    real_after_match = re.search(r"Real Person Characters.*?After Death:.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    
    if real_before_match:
        stats['real_before_abstain'] = float(real_before_match.group(1))
        stats['real_before_answer'] = float(real_before_match.group(2))
        stats['real_before_accuracy'] = float(real_before_match.group(3))
    if real_after_match:
        stats['real_after_abstain'] = float(real_after_match.group(1))
        stats['real_after_answer'] = float(real_after_match.group(2))
        stats['real_after_accuracy'] = float(real_after_match.group(3))
    
    # Extract mix fictional metrics
    mix_before_match = re.search(r"Mix Fictional Characters.*?Before Death:.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    mix_after_match = re.search(r"Mix Fictional Characters.*?After Death:.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    
    if mix_before_match:
        stats['mix_before_abstain'] = float(mix_before_match.group(1))
        stats['mix_before_answer'] = float(mix_before_match.group(2))
        stats['mix_before_accuracy'] = float(mix_before_match.group(3))
    if mix_after_match:
        stats['mix_after_abstain'] = float(mix_after_match.group(1))
        stats['mix_after_answer'] = float(mix_after_match.group(2))
        stats['mix_after_accuracy'] = float(mix_after_match.group(3))
    
    # Extract aggregated metrics
    aggregated_before_match = re.search(r"Aggregated Real Person \+ Mix Fictional.*?Before Death:.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    aggregated_after_match = re.search(r"Aggregated Real Person \+ Mix Fictional.*?After Death:.*?Abstain Rate: ([\d.]+).*?Answer Rate: ([\d.]+).*?Conditional Accuracy: ([\d.]+)", content, re.DOTALL)
    
    if aggregated_before_match:
        stats['aggregated_before_abstain'] = float(aggregated_before_match.group(1))
        stats['aggregated_before_answer'] = float(aggregated_before_match.group(2))
        stats['aggregated_before_accuracy'] = float(aggregated_before_match.group(3))
    if aggregated_after_match:
        stats['aggregated_after_abstain'] = float(aggregated_after_match.group(1))
        stats['aggregated_after_answer'] = float(aggregated_after_match.group(2))
        stats['aggregated_after_accuracy'] = float(aggregated_after_match.group(3))
    
    return stats

def plot_metrics(filepath, character_type='real', metric='abstain'):
    """
    Create a plot comparing metrics before and after death.
    
    Args:
        filepath: Path to the statistics file
        character_type: Type of character to plot ('real', 'mix', 'fully', or 'aggregated')
        metric: Metric to plot ('abstain', 'answer', 'accuracy')
    """
    stats = parse_stat_file(filepath)
    
    # Define character type data with all metrics
    character_types = {
        'real': {
            'name': 'Real Person',
            'abstain_before': stats.get('real_before_abstain', 0),
            'abstain_after': stats.get('real_after_abstain', 0),
            'answer_before': stats.get('real_before_answer', 0),
            'answer_after': stats.get('real_after_answer', 0),
            'accuracy_before': stats.get('real_before_accuracy', 0),
            'accuracy_after': stats.get('real_after_accuracy', 0)
        },
        'mix': {
            'name': 'Mix Fictional',
            'abstain_before': stats.get('mix_before_abstain', 0),
            'abstain_after': stats.get('mix_after_abstain', 0),
            'answer_before': stats.get('mix_before_answer', 0),
            'answer_after': stats.get('mix_after_answer', 0),
            'accuracy_before': stats.get('mix_before_accuracy', 0),
            'accuracy_after': stats.get('mix_after_accuracy', 0)
        },
        'fully': {
            'name': 'Fully Fictional',
            'abstain_before': stats.get('fully_fictional_abstain', 0),
            'abstain_after': stats.get('fully_fictional_abstain', 0),  # Same value
            'answer_before': stats.get('fully_fictional_answer', 0),
            'answer_after': stats.get('fully_fictional_answer', 0),    # Same value
            'accuracy_before': stats.get('fully_fictional_accuracy', 0),
            'accuracy_after': stats.get('fully_fictional_accuracy', 0) # Same value
        },
        'aggregated': {
            'name': 'Aggregated Real+Mix',
            'abstain_before': stats.get('aggregated_before_abstain', 0),
            'abstain_after': stats.get('aggregated_after_abstain', 0),
            'answer_before': stats.get('aggregated_before_answer', 0),
            'answer_after': stats.get('aggregated_after_answer', 0),
            'accuracy_before': stats.get('aggregated_before_accuracy', 0),
            'accuracy_after': stats.get('aggregated_after_accuracy', 0)
        }
    }
    
    # Get selected character type data
    if character_type not in character_types:
        raise ValueError(f"Invalid character type: {character_type}. Must be 'real', 'mix', 'fully', or 'aggregated'.")
    
    char_data = character_types[character_type]
    
    # Colors (previous version)
    colors = {
        'role_play': '#3d88ad',      # Blue (Role-play)
        'non_role_play': '#f2b56e',  # Orange/yellow (Non-role-play)
        'desirable': '#8ab07c'       # Green (Expected)
    }
    
    # Create figure with a specific layout for square plot
    fig = plt.figure(figsize=(18, 12))
    
    # Create a square plot area
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.85])  # Left, bottom, width, height in figure fraction
    
    # Define width and positions
    width = 0.3
    x1 = 0  # Position for "Before Death" group
    x2 = 1  # Position for "After Death" group
    
    # Get the specific metrics based on selected metric
    before_value = char_data[f'{metric}_before'] * 100  # Convert to percentage
    after_value = char_data[f'{metric}_after'] * 100    # Convert to percentage
    
    # Set default values based on the metric
    if metric == 'abstain':
        # For abstain metric
        non_role_play_before = 0
        desirable_before = 0
        desirable_after = 100  # Now 100%
    elif metric in ['answer', 'accuracy']:
        # For answer and accuracy metrics
        non_role_play_before = 100  # Now 100%
        desirable_before = 100      # Now 100%
        desirable_after = 0
    
    # Before death group
    ax.bar(x1 - width, before_value, width, label='Role-play', color=colors['role_play'])
    ax.bar(x1, non_role_play_before, width, label='Non-role-play', color=colors['non_role_play'])
    ax.bar(x1 + width, desirable_before, width, label='Expected', color=colors['desirable'])
    
    # After death group - maintain the same order
    ax.bar(x2 - width, after_value, width, color=colors['role_play'])
    ax.bar(x2, non_role_play_before, width, color=colors['non_role_play'])  # Use same value as before death
    ax.bar(x2 + width, desirable_after, width, color=colors['desirable'])
    
    # Add text annotations for all bars
    text_offset = 3  # Offset for text above bars
    
    # Before death group annotations
    ax.text(x1 - width, before_value + text_offset, f'{before_value:.1f}', ha='center')
    ax.text(x1, non_role_play_before + text_offset, f'{non_role_play_before:.1f}', ha='center')
    ax.text(x1 + width, desirable_before + text_offset, f'{desirable_before:.1f}', ha='center')
    
    # After death group annotations
    ax.text(x2 - width, after_value + text_offset, f'{after_value:.1f}', ha='center')
    ax.text(x2, non_role_play_before + text_offset, f'{non_role_play_before:.1f}', ha='center')
    ax.text(x2 + width, desirable_after + text_offset, f'{desirable_after:.1f}', ha='center')
    
    # Set plot properties
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(['Before Death', 'After Death'])
    
    # Set title based on metric
    metric_name = {'abstain': 'Abstain Rate', 'answer': 'Answer Rate', 'accuracy': 'Conditional Accuracy'}
    ax.set_ylabel(metric_name[metric] + ' (%)')  # Add percentage indication
    ax.set_ylim(0, 115)  # Update y-axis limit for percentages
    # if 'Aggregated' in char_data['name']:
    #     ax.set_title(f"Role-Play {metric_name[metric]}")
    # else:
    #     ax.set_title(f"{char_data['name']} {metric_name[metric]}")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=colors['role_play'], lw=4),
        Line2D([0], [0], color=colors['non_role_play'], lw=4),
        Line2D([0], [0], color=colors['desirable'], lw=4)
    ]
    
    # Add legend with appropriate labels
    labels = ['Role-play', 'Non-role-play', 'Expected']
    ax.legend(custom_lines, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Don't use tight_layout since we're manually positioning
    filename = f'plot/{args.experiment_type}/{args.model_id}/{args.type}/{args.question_type}/{metric}_rates_{character_type}.pdf'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename,  bbox_inches='tight', format='pdf')
    # plt.show()
    print(f"Plot saved as {filename}")

if __name__ == "__main__":
    filepath = f'evaluation/{args.experiment_type}/{args.model_id}/{args.type}/{args.question_type}/stat_{args.type}_{args.question_type}.txt'
    output_dir = f'plot/{args.experiment_type}/{args.model_id}/{args.type}/{args.question_type}'
    os.makedirs(output_dir, exist_ok=True)
    plot_metrics(filepath, args.fictional_type, args.metric)