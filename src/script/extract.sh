types=("normal_temp1" "normal_temp2" "normal_temp3")
question_types=("four_president")
model_id=("llama")
exp=("add_prompt")


for type in "${types[@]}"; do
    for question in "${question_types[@]}"; do
        for model_id in "${model_id[@]}"; do
            for exp in "${exp[@]}"; do
                python src/plot_prep/extract_evaluation.py --type $type --question_type $question --model_id $model_id --experiment_type $exp
            done
        done
    done
done