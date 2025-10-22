question_types=("four_president")
types=("die_yearly_cot")
model_id=("gemma")
exp=("add_prompt")

for question_type in "${question_types[@]}"; do
    for type in "${types[@]}"; do
        for model_id in "${model_id[@]}"; do
            echo "Running with question_type=$question_type, type=$type, exp=$exp"
            python src/eval/accuracy_label.py --type $type --question_type $question_type --model_id $model_id --experiment_type $exp
        done
    done
done
