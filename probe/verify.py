import pickle
import os
import numpy as np
import torch
from pprint import pprint
from transformer_lens import HookedTransformer


def load_model(model_id):
    model = HookedTransformer.from_pretrained(model_id)
    tokenizer = model.tokenizer
    return model, tokenizer

def load_and_decode_pickle(pickle_path):
    """
    Load a pickle file and decode the first entry.
    
    Args:
        pickle_path (str): Path to the pickle file
    """
    print(f"Loading data from {pickle_path}")

    model, tokenizer = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # Check if file exists
    if not os.path.exists(pickle_path):
        print(f"Error: File {pickle_path} does not exist")
        return
    
    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Print basic info about the loaded data
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys in data: {list(data.keys())}")
        first_key = list(data.keys())[0]
        print(f"\nFirst entry (key: {first_key}):")
        pprint(data[first_key])
    elif isinstance(data, list) or isinstance(data, tuple):
        print(f"Number of entries: {len(data)}")
        print(f"\nFirst entry:")
        pprint(data[0])
        breakpoint()
        print(tokenizer.decode(data[0]['prompt']))
    else:
        print(f"\nData content:")
        pprint(data)

if __name__ == "__main__":
    # Update this path to the location of your pickle file
    pickle_path = "death_probe/data/normal_overall_death/model_probe_train_death.pkl"
    
    load_and_decode_pickle(pickle_path)
