#!/bin/bash

MODEL_FAMILY="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_FAMILY="google/gemma-2-9b-it"

#normal , new_normal, death
prompt_type="real_art_death_current"
activation_output_type="real_art_death_current_activation_datasets"
probe_output_type="real_art_death_current"
entity_type="art"


python save_activations.py --model $MODEL_FAMILY --entity_type $entity_type --prompt_type $prompt_type --output_type $activation_output_type --use_tl

python probe_experiment.py --experiment_name $entity_type --model $MODEL_FAMILY --entity_type $entity_type --feature_name release_date --prompt_name release --output_type $probe_output_type
