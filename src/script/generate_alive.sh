model_ids=("llama" "gemma")

for model_id in "${model_ids[@]}"; do
    python src/generation/alive_generation.py --model_id $model_id
done