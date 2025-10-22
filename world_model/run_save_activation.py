import subprocess

# Define the model and entity type
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "google/gemma-2-9b-it"
entity_type = "presidential"

# Command to run the save_activations.py script
#daeth
command = [
    "python", "save_activations.py",
    "--model", model_name,
    "--entity_type", entity_type,
    "--use_tl",
    "--prompt_type", "_normal",
    "--output_type", "presidential_normal_activation_datasets"
]

# Run the script
try:
    subprocess.run(command, check=True)
    print("Script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the script: {e}")