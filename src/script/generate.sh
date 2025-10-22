#!/bin/bash

question_types=("four_president")
types=("normal_temp4")
model_id=("llama")
exp=("add_prompt")

for exp in "${exp[@]}"; do
    for question_type in "${question_types[@]}"; do
        for type in "${types[@]}"; do
            echo "Running with question_type=$question_type, type=$type, exp=$exp"
            python src/generation/generate.py --question_type $question_type --type $type --model_id $model_id --exp $exp --continue_from_file False
        done
    done
done
