export question1=four_president_given_death
export question2=yearly_given_death
export type1=die_yearly
export type2=die_yearly
export model_id=llama
mkdir -p evaluation/full/${model_id}/${type1}/combined
python src/plot_prep/combination.py --yearly evaluation/full/${model_id}/${type1}/${question1}/final_evaluation_${type1}_${question1}.json --president evaluation/full/${model_id}/${type2}/${question2}/final_evaluation_${type2}_${question2}.json --output evaluation/full/${model_id}/${type2}/combined/final_evaluation_${type2}_combined.json