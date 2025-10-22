for i in {1..30}
do
python data_process.py --model_id meta-llama/Llama-3.1-8B-Instruct --death_year_idx $i --input_data_path death_probe/data/labeled_with_death_year_cleaned.json --output_name probe_death_idx
done
