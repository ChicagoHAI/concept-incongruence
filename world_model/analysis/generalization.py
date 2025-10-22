import pandas as pd
import numpy as np
from probe_experiment import load_probe_results


def make_result_df(models, prompts, experiment_name, entity_type, feature_name):
    rdfs = {}
    for model in models:
        for prompt in prompts:
            probe_result = load_probe_results(
                experiment_name, model, entity_type, feature_name, prompt)
            rdf = pd.DataFrame(probe_result['scores']).T
            rdf.index.name = 'layer'
            # rdf = rdf.reset_index()
            rdfs[model, prompt] = rdf

    rdf = pd.concat(rdfs, names=['model', 'prompt'])
    rdf.columns = rdf.columns.map('_'.join)
    rdf = rdf.reset_index()
    return rdf


def make_probe_direction_matrices(models, prompts, experiment_name, entity_type, feature_type):
    probe_mats = {}
    for model in models:
        for prompt in prompts:
            probe_result = load_probe_results(
                experiment_name, model, entity_type, feature_type, prompt)
            model_directions = np.stack(
                list(probe_result['probe_directions'].values()))
            probe_mats[model, prompt] = model_directions
    return probe_mats


# OLD
def make_generalization_rdf(probe_eval):
    generalization_results = {}
    for probe_train_key, result_dict in probe_eval.items():
        for probe_test_data_key, result in result_dict.items():
            generalization_results[probe_train_key +
                                   probe_test_data_key] = result
    rdf = pd.DataFrame(generalization_results).T
    rdf.index.names = ['train_layer',
                       'train_prompt', 'test_layer', 'test_prompt']
    return rdf

