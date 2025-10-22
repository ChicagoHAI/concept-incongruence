# Define arrays for the variables you want to loop over
types=("normal_temp1" "normal_temp2" "normal_temp3")
questions=("four_president")
model_ids=("llama")
experiment_type=("add_prompt")

# Loop through all combinations
for type in "${types[@]}"; do
    for question in "${questions[@]}"; do
        for model_id in "${model_ids[@]}"; do
            # Set the save directory
            SAVE="evaluation/${experiment_type}/${model_id}/${type}/${question}"
            
            # Create directory if it doesn't exist
            mkdir -p ${SAVE}
            
            # Run the Python script
            python src/plot_prep/stat_plot.py --type $type --question_type $question --model_id $model_id --experiment_type $experiment_type > ${SAVE}/stat_${type}_${question}.txt
            
            echo "Completed processing for type=$type, question=$question, model_id=$model_id"
        done
    done
done