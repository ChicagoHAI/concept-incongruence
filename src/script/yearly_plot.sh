#!/bin/bash

# Arrays of parameters to loop through
types=("die_yearly")  # Add or modify types as needed
questions=("yearly_given_death")  # Add or modify question types as needed
models=("llama")  # Add or modify models as needed
plot_type=("results" "ideal")
exp=("full")

# Nested loops to process all combinations
for type in "${types[@]}"; do
    for question in "${questions[@]}"; do
        for model_id in "${models[@]}"; do
            for plot_type in "${plot_type[@]}"; do
                echo "Processing: type=$type, question=$question, model=$model_id, plot_type=$plot_type"
                python src/plot_prep/stat_yearly.py --type $type --question_type $question --model_id $model_id --plot_type $plot_type --experiment_type $exp
            done
        done
    done
done
