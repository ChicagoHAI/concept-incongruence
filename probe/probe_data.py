#training a probe on the role dataset
import os
import torch
import einops
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import datasets
from tqdm import trange
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, StoppingCriteriaList, LogitsProcessorList, LogitsProcessor, StoppingCriteria
import argparse

# DATA_PATH = 'Persona_Understanding/role/data'
DATA_PATH = 'concept/gemma_death_probe/data'
parser = argparse.ArgumentParser()
parser.add_argument("--train_on", type=str, default="death")
parser.add_argument("--type", type=str, default="death_idx")
args = parser.parse_args()

device = "cuda"
# D_MODEL = 3584
D_MODEL=4096
TRAIN_ON = "fictional"
PROBE_PATH = "concept/gemma_death_probe/ai_death/probe_31_-1"
os.makedirs(PROBE_PATH, exist_ok=True)
if TRAIN_ON == "death" or TRAIN_ON == "fictional":
    N_CLASSES = 2



def load_model(model_id):
    model = HookedTransformer.from_pretrained(model_id)
    tokenizer = model.tokenizer
    return model, tokenizer
    
class LinearLayer(nn.Module):
    def __init__(self, dim=D_MODEL, out=N_CLASSES):
        super().__init__()
        self.layer1 = nn.Linear(dim, out)
        
    def forward(self, x):
        # If x has an extra dimension, flatten it
        if len(x.shape) == 3:  # If shape is [batch, 1, dim]
            x = x.squeeze(1)   # Make it [batch, dim]
        y = self.layer1(x)
        return y

class MLP(nn.Module):
    def __init__(self, dim=D_MODEL, hidden_dim=100, out=N_CLASSES):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out)
        self.act = nn.ReLU()
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.act(y)
        y = self.layer2(y)
        return y

def load_data(train_path, valid_path):
    train_df = pd.read_pickle(train_path)
    valid_df = pd.read_pickle(valid_path)
    return train_df, valid_df

def get_hidden_states(model, x, layer=-1, extract_position=-1):
    with torch.inference_mode():
        _, cache = model.run_with_cache(x, return_type = None)
        resid_post = cache["resid_post", layer]
        #Dim: [batch, seq_len, hidden_dim]
        hs = resid_post[:,extract_position,:]
        return hs

def get_tensors(df, model, tokenizer, layer=-1, extract_position=-1):
    x = []
    y = []

    # Process each row in the dataframe
    for row in df:
        # Get hidden states
        hidden_state = get_hidden_states(model, row['prompt'], layer=layer, extract_position=extract_position)
        x.append(hidden_state)
        y.append(row['label'])
        
    x = torch.cat(x, dim=0)  # Using torch.cat instead of np.concatenate
    y = torch.tensor(y)
    
    return x, y

def train_probe(x_train, y_train, probe, optimizer, n_epochs=100, batch_size = 50, log=False, arch='linear'):
    
    # Add validation check
    n_classes = probe.layer1.out_features
    min_loss = 99999
    if y_train.max() >= n_classes or y_train.min() < 0:
        raise ValueError(f"Labels must be in range [0, {n_classes-1}], but got range [{y_train.min()}, {y_train.max()}]")
    
    n_epochs = n_epochs
    batches_per_epoch = len(x_train) // batch_size
    losses = []
    loss_fn = nn.CrossEntropyLoss()

    for epoch in trange(n_epochs):

        for i in range(batches_per_epoch):
            start = i * batch_size

            x_batch = x_train[start:start+batch_size].to(device)
            y_batch = y_train[start:start+batch_size].to(device)

            y_pred = probe.forward(x_batch)
            # breakpoint()
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        
        if log:
            print("Avg. loss in epoch {}: {}".format(epoch, np.mean(losses)))
        # breakpoint()
        if losses[-1] < min_loss:
            min_loss = losses[-1]
            best_probe = probe
            torch.save(best_probe.state_dict(), os.path.join(PROBE_PATH, f"best_probe_year_{arch}_{lr}_{batch_size}_{epochs}.pth"))
    
    return probe, losses[-1]

def create_probe(x_train, y_train, arch='linear',  epochs=200, log=True,lr=0.001, batch_size=50):
    assert arch in model_mapping, "Please redefine your model"
    
    if arch == 'linear':
        probe = LinearLayer().to(device)
    elif arch == 'mlp':
        probe = MLP().to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    
    probe, loss = train_probe(x_train, y_train, probe, optimizer, epochs, batch_size,log=False, arch=arch)
    
    print(f"Finished training with loss = {loss}")
    
    return probe,loss

def evaluate(probe, x_eval, y_eval):
    
    with torch.no_grad():
        predictions = probe.forward(x_eval.to(device)).detach().cpu()
        predictions = predictions.argmax(-1)
    
    accuracy = (predictions == y_eval).sum() / len(y_eval)
    print(f" Accuracy: {accuracy}")
    
    return accuracy, predictions



if __name__ == "__main__":
    model, tokenizer = load_model("meta-llama/Llama-3.1-8B-Instruct")
    lr = 0.001
    batch_size = 50
    epochs = 100
    idx_list = list(range(1,31))
    year_acc = []
    year_loss = []

    if args.type == "death_idx":
        file_name = f"probe_{args.type}"
    
    for i in idx_list:
        train_df = pd.read_pickle(f'concept/llama_death_probe/data/rp_year_task/probe_death_idx_{i}_train_death.pkl')
        valid_df = pd.read_pickle(f'concept/llama_death_probe/data/rp_year_task/probe_death_idx_{i}_valid_death.pkl')


        layer_pos_acc =[]
        layer_pos_loss = []
        pos_list = [-1]
        layer_list = [31]
        for layer in layer_list:
            pos_acc = []
            pos_loss = []
            for pos in pos_list:
                print(f"Testing position {pos} in layer {layer}")
                x_train, y_train = get_tensors(train_df, model, tokenizer,layer=layer, extract_position=pos)
                x_eval, y_eval = get_tensors(valid_df, model, tokenizer,layer=layer, extract_position=pos)

                model_mapping = {"linear": LinearLayer, "mlp": MLP}
                probe, loss = create_probe(x_train, y_train, arch='mlp', epochs=epochs, log=True, lr=lr, batch_size=batch_size)
                acc, pred = evaluate(probe, x_eval, y_eval)
                pos_acc.append(acc)
                pos_loss.append(loss)
            layer_pos_acc.append(pos_acc)
            layer_pos_loss.append(pos_loss)
        year_acc.append(layer_pos_acc)
        year_loss.append(layer_pos_loss)

        print(year_acc)
        
    # os.makedirs(f"death_probe/results", exist_ok=True)
    # with open(f"death_probe/results/probe_ai_year_task_padleft.txt", "w") as f:
    #     f.write(f"Accuracy: {year_acc}\n")
    #     f.write(f"Loss: {year_loss}\n")