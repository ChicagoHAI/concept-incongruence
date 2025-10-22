#!/bin/bash

# Arrays of options for each parameter
types=("die_yearly")
questions=("four_president_given_death")
model_ids=("llama" )
fictional_types=("aggregated") #aggregated
metrics=("abstain" "answer" "accuracy")
experiment_type=full

# Loop through all combinations
for type in "${types[@]}"; do
    for question in "${questions[@]}"; do
        for model_id in "${model_ids[@]}"; do
            for fictional_type in "${fictional_types[@]}"; do
                for metric in "${metrics[@]}"; do
                    echo "Running with: type=$type, question=$question, model_id=$model_id, fictional_type=$fictional_type, metric=$metric"
                    
                    python plot/subplots.py --type "$type" --question_type "$question" --model_id "$model_id" --fictional_type "$fictional_type" --metric "$metric" --experiment_type $experiment_type
                    
                    echo "----------------------------------------"
                done
            done
        done
    done
done