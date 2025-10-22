import json
import pickle
from transformer_lens import HookedTransformer
import random
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--death_year_idx", type=int, default=0)
parser.add_argument("--input_data_path", type=str, default="death_probe/death_new.json")
parser.add_argument("--output_name", type=str, default="model_probe_die")
args = parser.parse_args()


death_idx = args.death_year_idx
model_id = args.model_id
input_data_path = args.input_data_path
output_name = args.output_name


def load_model(model_id):
    model = HookedTransformer.from_pretrained(model_id)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    return model, tokenizer

new_data = []
with open(input_data_path, 'r') as f:
    data = json.load(f)

model, tokenizer = load_model(model_id)

# Count labels while creating new_data
label_0_count = 0
label_1_count = 0

# DO you know Harry Potter? 99% accuracy check that
die = "You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime or within their story. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. Please only output the answer to the questions"
normal = "You should play the role given to you. Please only output the answer to the questions."
ai = "You are a helpful assistant."
#please only output the answer to the questions
for item in tqdm(data, desc="Processing data"):
#     #overall_die. normal/die + you are ...
#     #ooverall_ai: ai + char
#     #ai_death_yn: Is the character {item['name']} in {int(item['death_year']) - death_idx} dead or alive?

    before_input_prompt = normal + f" Tell me something about {item['name']} in {int(item['death_year']) - death_idx}."
    before_output = tokenizer(before_input_prompt, return_tensors="pt", padding='max_length',max_length=100).input_ids
    
    if item["label"] == 0:
        label_0_count += 1
    else:
        label_1_count += 1
        
    new_data.append({
        "prompt": before_output,
        "label": 0
    })
    
    after_input_prompt = normal + f" Tell me something about {item['name']} in {int(item['death_year']) + death_idx}."
    after_output = tokenizer(after_input_prompt, return_tensors="pt", padding='max_length',max_length=100).input_ids
    
    new_data.append({
        "prompt": after_output,
        "label": 1
    })
    # input_prompt =  f" Tell me something about {item['name']}."
    # output = tokenizer(input_prompt, return_tensors="pt", padding='max_length',max_length=100).input_ids
    # new_data.append({
    #     "prompt": output,
    #     "label": item["isDead"]
    # })
    # if item["label"] == 0:
    #     label_0_count += 1
    # else:
    #     label_1_count += 1

print(f"Original distribution - Label 0: {label_0_count}, Label 1: {label_1_count}")

# Split data by labels first
label_0_items = [item for item in new_data if item["label"] == 0]
label_1_items = [item for item in new_data if item["label"] == 1]

# Calculate split sizes for each label
split_idx_0 = int(len(label_0_items) * 0.8)
split_idx_1 = int(len(label_1_items) * 0.8)

# Split into train and valid while maintaining label separation
train_0 = label_0_items[:split_idx_0]
train_1 = label_1_items[:split_idx_1]
valid_0 = label_0_items[split_idx_0:]
valid_1 = label_1_items[split_idx_1:]

# Balance train data
train_target_size = min(len(train_0), len(train_1))
train_data = (
    random.sample(train_0, train_target_size) +
    random.sample(train_1, train_target_size)
)

# Balance valid data
valid_target_size = min(len(valid_0), len(valid_1))
valid_data = (
    random.sample(valid_0, valid_target_size) +
    random.sample(valid_1, valid_target_size)
)

# Shuffle both sets
random.shuffle(train_data)
random.shuffle(valid_data)

print(f"Balanced train dataset size: {len(train_data)} (Label 0: {train_target_size}, Label 1: {train_target_size})")
print(f"Balanced valid dataset size: {len(valid_data)} (Label 0: {valid_target_size}, Label 1: {valid_target_size})")

output_dir = "/net/scratch2/smallyan/concept/llama_death_probe/data/rp_year_task"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/{output_name}_{death_idx}_train_death.pkl", "wb") as f:
    pickle.dump(train_data, f)

with open(f"{output_dir}/{output_name}_{death_idx}_valid_death.pkl", "wb") as f:
    pickle.dump(valid_data, f)
