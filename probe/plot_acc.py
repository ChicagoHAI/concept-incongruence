import matplotlib.pyplot as plt
import seaborn as sns

OUT_PATH = 'training_output'

def extract_accuracies(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Accuracy" in line:
                # Extract the accuracy value from the line
                parts = line.split(':')
                if len(parts) > 1:
                    accuracy = float(parts[1].strip()) * 100  # Convert to percentage
                    accuracies.append(accuracy)
    return accuracies

# Paths to the files
# probe_out_path = 'probe.out' # fictionality
probe_death_out_path = f'{OUT_PATH}/new_gemma/gemma_rp_overall_linear.out' 
probe_overall_ai_out_path = f'{OUT_PATH}/new_gemma/gemma_nrp_overall_linear.out' 
# Extract accuracies
probe_death_accuracies = extract_accuracies(probe_death_out_path)
probe_overall_ai_accuracies = extract_accuracies(probe_overall_ai_out_path)


# Set font size for various elements
sns.set_theme(style="white")
sns.set_context("talk", font_scale=2.5)

# Plot the results using Seaborn
fig = plt.figure(figsize=(18, 12))
    
    # Create a square plot area
ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Left, bottom, width, height in figure fraction

metric_colors = {'Role-Play': '#1f77b4', 'Non-Role-Play': '#ff7f0e', 'accuracy': '#2ca02c'} 
# sns.lineplot(data=probe_accuracies, label='Probe Fictionality Accuracies', marker='o')
plt.plot(probe_death_accuracies, label='Role-Play', marker='D', linewidth=5, markersize=10, color=metric_colors['Role-Play'])
# sns.lineplot(data=probe_restricted_death_accuracies, label='Restricted', marker='s')
plt.plot(probe_overall_ai_accuracies, label='Non-Role-Play', marker='s', linewidth=5, markersize=10, color=metric_colors['Non-Role-Play'])
# sns.lineplot(data=probe_overall_role_play_accuracies, label='Role-Play', marker='D')

# plt.title('Layer-wise Role-Play Probe Test Accuracies')
plt.xlabel('Layer')
plt.ylabel('Accuracy (%)')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.grid(False)
plt.ylim(50, 105)  # Set y-axis range from 50% to 100%
plt.tight_layout()  # Adjusts the plot to make room for the legend
plt.savefig('probe_acc_overall_linear_gemma.pdf', format='pdf', bbox_inches='tight')  # Ensures the legend is included in the saved figure
