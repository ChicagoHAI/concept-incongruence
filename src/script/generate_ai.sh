#!/bin/bash

question_types=("commonsense")
types=("QA")
model_ids=("llama" "gemma")

for question_type in "${question_types[@]}"; do
    for type in "${types[@]}"; do
        for model_id in "${model_ids[@]}"; do
            echo "Running with question_type=$question_type, type=$type"
            python src/generation/ai_generation.py --question_type $question_type --type $type --model_id $model_id
        done
    done
done
