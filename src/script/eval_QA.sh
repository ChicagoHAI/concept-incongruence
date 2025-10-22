model_ids=("gemma")
question_types=("commonsense")
exp=("QA_temp3")
type="normal_QA"

for model_id in "${model_ids[@]}"; do
    for question_type in "${question_types[@]}"; do
        for exp in "${exp[@]}"; do
            echo "Running with model_id=$model_id, question_type=$question_type, type=$type, exp=$exp"
            python src/eval/eval_qa_acc.py --model_id $model_id --question_type $question_type --experiment_type $exp --type $type
        done
    done
done