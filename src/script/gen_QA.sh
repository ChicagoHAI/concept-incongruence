#!/bin/bash

question_types=("four_president")
types=("normal_temp2")
model_id=("llama" "gemma")
exp=("add_prompt")

for exp in "${exp[@]}"; do
    for question_type in "${question_types[@]}"; do
        for type in "${types[@]}"; do
            echo "Running with question_type=$question_type, type=$type, exp=$exp"
            python src/generation/generate.py --question_type $question_type --type $type --model_id $model_id --exp $exp --continue_from_file False
        done
    done
done


# question_types=("commonsense")
# types=("normal_QA")
# model_id=("llama")
# exp=("add_prompt")

# for exp in "${exp[@]}"; do
#     for question_type in "${question_types[@]}"; do
#         for type in "${types[@]}"; do
#             echo "Running with question_type=$question_type, type=$type, exp=$exp"
#             python src/generation/generation_QA.py --question_type $question_type --type $type --model_id $model_id --exp $exp --continue_from_file False
#         done
#     done
# done
