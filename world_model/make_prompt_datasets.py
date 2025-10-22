import argparse
from feature_datasets import time_art, common, space_us, space_world, headline, space_nyc, historical, custom_questions
from transformers import AutoTokenizer
from functools import partial


ENTITY_PROMPTS = {
    # time
    'art': time_art.ART_PROMPTS,
    'headline': headline.HEADLINE_PROMPTS,
    'historical_figure': historical.HISTORICAL_PROMPTS,
    'presidential': custom_questions.PRESIDENT_PROMPTS,
    'presidential_all': custom_questions.PRESIDENT_PROMPTS,
    'presidential_baseline': custom_questions.PRESIDENT_PROMPTS,
    'presidential_real': custom_questions.PRESIDENT_PROMPTS,
    'presidential_baseline_prompt': custom_questions.PRESIDENT_PROMPTS,

    # space
    'world_place': space_world.PLACE_PROMPTS,
    'us_place': space_us.US_PLACE_PROMPTS,
    'nyc_place': space_nyc.NYC_PLACE_PROMPTS,
}

DATASET_FUNCTIONS = {
    'art': time_art.make_art_prompt_dataset,
    'headline': headline.make_headline_prompt_dataset,
    'historical_figure': historical.make_historical_figure_prompt_dataset,
    'presidential': custom_questions.make_custom_question_dataset,
    'presidential_all': custom_questions.make_custom_question_dataset,
    'presidential_baseline': custom_questions.make_custom_question_dataset,
    'presidential_real': custom_questions.make_custom_question_dataset,
    'presidential_baseline_prompt': custom_questions.make_custom_question_dataset,
    
    'world_place': partial(space_world.make_world_prompt_dataset, entity_col='name'),
    'us_place': partial(common.make_prompt_dataset, entity_col='name'),
    'nyc_place': partial(space_nyc.make_nyc_prompt_dataset, entity_col='name'),
}


def make_and_save_tokenized_datasets(
        tokenizer, model_family, entity_type, prompt_dict, ds_make_fn, prompt_type):
    entity_data = common.load_entity_data(entity_type)

    for short_prompt, full_prompt in prompt_dict.items():
        dataset = ds_make_fn(
            short_prompt, full_prompt, tokenizer, entity_data)

        save_path = common.prompt_data_path(
            entity_type, short_prompt, model_family, prompt_type)
        dataset.save_to_disk(save_path)


def load_tokenizer(model_name):
    if 'Llama-2' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif 'pythia' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif 'Llama-3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif 'gemma' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise ValueError('invalid model name')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='pythia')
    parser.add_argument('--entity_type', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='_new_normal')

    args = parser.parse_args()
    model_family = args.model_family
    tokenizer = load_tokenizer(model_family)

    if args.entity_type == 'all':
        for name, dataset_fn in DATASET_FUNCTIONS.items():
            prompt_dict = ENTITY_PROMPTS[name]
            make_and_save_tokenized_datasets(
                tokenizer, model_family, name, prompt_dict, dataset_fn, args.prompt_type)
    elif args.entity_type in DATASET_FUNCTIONS:
        dataset_fn = DATASET_FUNCTIONS[args.entity_type]
        prompt_dict = ENTITY_PROMPTS[args.entity_type]
        make_and_save_tokenized_datasets(
            tokenizer, model_family, args.entity_type, prompt_dict, dataset_fn, args.prompt_type)
    else:
        raise ValueError('dataset not found')
