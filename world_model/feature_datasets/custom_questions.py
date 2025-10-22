import os
import json
import random
import torch
import pandas as pd
import datasets
from .common import *
from typing import Dict, List, Optional
import csv

PRESIDENT_PROMPTS = {
    'president_normal': 'nothing'
}

def make_questions():
    questions = []
    test_range = range(1, 46)
    
    # Presidential term data (start year, end year)
    presidential_terms = {
        1: ("1789", "1797"),  # Washington
        2: ("1797", "1801"),  # Adams
        3: ("1801", "1809"),  # Jefferson
        4: ("1809", "1817"),  # Madison
        5: ("1817", "1825"),  # Monroe
        6: ("1825", "1829"),  # J.Q. Adams
        7: ("1829", "1837"),  # Jackson
        8: ("1837", "1841"),  # Van Buren
        9: ("1841", "1841"),  # W.H. Harrison
        10: ("1841", "1845"),  # Tyler
        11: ("1845", "1849"),  # Polk
        12: ("1849", "1850"),  # Taylor
        13: ("1850", "1853"),  # Fillmore
        14: ("1853", "1857"),  # Pierce
        15: ("1857", "1861"),  # Buchanan
        16: ("1861", "1865"),  # Lincoln
        17: ("1865", "1869"),  # A. Johnson
        18: ("1869", "1877"),  # Grant
        19: ("1877", "1881"),  # Hayes
        20: ("1881", "1881"),  # Garfield
        21: ("1881", "1885"),  # Arthur
        22: ("1885", "1889"),  # Cleveland (1st)
        23: ("1889", "1893"),  # B. Harrison
        24: ("1893", "1897"),  # Cleveland (2nd)
        25: ("1897", "1901"),  # McKinley
        26: ("1901", "1909"),  # T. Roosevelt
        27: ("1909", "1913"),  # Taft
        28: ("1913", "1921"),  # Wilson
        29: ("1921", "1923"),  # Harding
        30: ("1923", "1929"),  # Coolidge
        31: ("1929", "1933"),  # Hoover
        32: ("1933", "1945"),  # F.D. Roosevelt
        33: ("1945", "1953"),  # Truman
        34: ("1953", "1961"),  # Eisenhower
        35: ("1961", "1963"),  # Kennedy
        36: ("1963", "1969"),  # L.B. Johnson
        37: ("1969", "1974"),  # Nixon
        38: ("1974", "1977"),  # Ford
        39: ("1977", "1981"),  # Carter
        40: ("1981", "1989"),  # Reagan
        41: ("1989", "1993"),  # G.H.W. Bush
        42: ("1993", "2001"),  # Clinton
        43: ("2001", "2009"),  # G.W. Bush
        44: ("2009", "2017"),  # Obama
        45: ("2017", "2021"),  # Trump
    }
    
    for i in test_range:
        # Standard start/end year questions
        # questions.append(f"What is the start year of the {i}th president of the United States?")
        # questions.append(f"What is the end year of the {i}th president of the United States?")
        
        # Questions for each year of the president's term
        if i in presidential_terms:
            start_year = int(presidential_terms[i][0])
            end_year = int(presidential_terms[i][1])
            for j, year in enumerate(range(start_year, end_year + 1), 1):
                ordinal = f"{j}{'st' if j == 1 else 'nd' if j == 2 else 'rd' if j == 3 else 'th'}"
                questions.append(f"What is the {ordinal} year of the {i}th president of the United States?")
    
    return questions

def load_character_data():
    # Path is relative to the project root
    character_path = 'data/full_question.json'
    
    with open(character_path, 'r') as f:
        character_data = json.load(f)
            
    # Extract characters from the nested structure
    real_people = []
    fictional_chars = []
    
    for category in character_data:
        if category['type'] == 'real_person':
            real_people.extend(category['characters'])
        # elif category['type'] == 'mix_fictional':
        #     fictional_chars.extend(category['characters'])
    
    # Sort both lists to get consistent results
    real_people.sort(key=lambda x: x['role'])
    # fictional_chars.sort(key=lambda x: x['role'])
    
    # Use the last 20 of each for test set (if available)
    test_real = real_people[-20:] if len(real_people) >= 20 else real_people
    # test_fictional = fictional_chars[-20:] if len(fictional_chars) >= 20 else fictional_chars
    # test_characters = test_real + test_fictional
    test_characters = test_real
    
    # Use the rest for training
    train_real = real_people[:-20] if len(real_people) >= 20 else []
    # train_fictional = fictional_chars[:-20] if len(fictional_chars) >= 20 else []
    # train_characters = train_real + train_fictional
    train_characters = train_real
    return test_characters, train_characters


def make_custom_question_dataset(short_prompt, prompt, tokenizer, president_df, use_character=True, sample_character=True):
    """
    Create a dataset from custom questions instead of artwork data.
    
    Args:
        short_prompt: Identifier for the prompt type
        prompt: The prompt template
        tokenizer: Tokenizer to use
        questions: List of question strings
        use_character: Whether to add character context to the prompts
    
    Returns:
        A dataset containing tokenized questions
    """
    
    questions = make_questions()
    dataset_strings = questions
    new_strings = []

    
    # If we're not using character mode, create a DataFrame from generated questions
    if not use_character:
        # Create a simple DataFrame with the questions
        questions_df = pd.DataFrame({
            'question': questions
        })
        dataset_strings = questions
        df_to_use = questions_df
    elif use_character and not sample_character:
        # Use the provided president_df with character information
        new_strings = []
        for _, row in president_df.iterrows():
            character = row.character
            year = row.death_year
            s = row.question
            if row.type == 'mix_fictional':
                character_description = f"{character} from {row.source}"
            new_strings.append(f" You should play the role given to you. Please only output the answer to the questions. You are {character}, {s}")
            print(new_strings[-1])
        dataset_strings = new_strings
        df_to_use = president_df
    elif use_character and sample_character:
        # Sample a random character from the president_df
        test_characters, train_characters = load_character_data()
        for i, s in enumerate(dataset_strings):
            if president_df.iloc[i].is_test:
                character = random.choice(test_characters)
            else:
                character = random.choice(train_characters)
            year=character.get('year', 2000)
            char_role = character['role']
                
            # Add source information for fictional characters
            character_description = char_role
            #Prompt
            new_strings.append(f" You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly. Please only output the answer to the questions. You died in {year}. This year is 2023.You are {character['role']}, {s}")
            print(new_strings[-1])
        print(len(new_strings))
        dataset_strings = new_strings
        df_to_use = president_df

    # Tokenize the strings
    token_ids = tokenizer.batch_encode_plus(
        dataset_strings,
        return_tensors='pt',
        padding=True,
        add_special_tokens=False,
        return_attention_mask=False
    )['input_ids']
    


    # Add bos token
    token_ids = torch.cat([
        torch.ones(token_ids.shape[0], 1,
                   dtype=torch.long) * tokenizer.bos_token_id,
        token_ids], dim=1
    )

    # Create masks for the questions
    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    question_mask = torch.ones_like(token_ids, dtype=torch.bool)
    question_mask[:, prompt_tokens] = False
    question_mask[token_ids == tokenizer.pad_token_id] = False

    # Create dataset
    dataset = datasets.Dataset.from_dict({
        'entity': df_to_use.question.values.tolist()[:len(token_ids)],
        'input_ids': token_ids.tolist(),
        'entity_mask': question_mask.tolist(),
    })

    dataset.set_format(type='torch', columns=['input_ids'])

    return dataset

class TemporalDataManager(EntityDataManager):
    def __init__(self, entity_type, prompt_dict):
        self.entity_type = entity_type
        self.prompt_dict = prompt_dict
        self.entity_data = None  # DataFrame loaded when needed

    def get_feature_values(self, feature_name):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        time = pd.to_datetime(
            self.entity_data[feature_name],
            format='%Y'
        )
        return time.values.astype(int)

    def make_and_save_tokenized_datasets(self, tokenizer, model_family, prompt_type):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        for short_prompt, full_prompt in self.prompt_dict.items():
            dataset = make_custom_question_dataset(
                full_prompt, tokenizer, self.entity_data)

            save_path = self.prompt_data_path(short_prompt, model_family, prompt_type)
            dataset.save_to_disk(save_path)

def create_presidential_questions_csv(output_file="data/entity_datasets/presidential_real.csv"):
    # Presidential term data (start year, end year)
    presidential_terms = {
        1: ("1789", "1797"),  # Washington
        2: ("1797", "1801"),  # Adams
        3: ("1801", "1809"),  # Jefferson
        4: ("1809", "1817"),  # Madison
        5: ("1817", "1825"),  # Monroe
        6: ("1825", "1829"),  # J.Q. Adams
        7: ("1829", "1837"),  # Jackson
        8: ("1837", "1841"),  # Van Buren
        9: ("1841", "1841"),  # W.H. Harrison
        10: ("1841", "1845"),  # Tyler
        11: ("1845", "1849"),  # Polk
        12: ("1849", "1850"),  # Taylor
        13: ("1850", "1853"),  # Fillmore
        14: ("1853", "1857"),  # Pierce
        15: ("1857", "1861"),  # Buchanan
        16: ("1861", "1865"),  # Lincoln
        17: ("1865", "1869"),  # A. Johnson
        18: ("1869", "1877"),  # Grant
        19: ("1877", "1881"),  # Hayes
        20: ("1881", "1881"),  # Garfield
        21: ("1881", "1885"),  # Arthur
        22: ("1885", "1889"),  # Cleveland (1st)
        23: ("1889", "1893"),  # B. Harrison
        24: ("1893", "1897"),  # Cleveland (2nd)
        25: ("1897", "1901"),  # McKinley
        26: ("1901", "1909"),  # T. Roosevelt
        27: ("1909", "1913"),  # Taft
        28: ("1913", "1921"),  # Wilson
        29: ("1921", "1923"),  # Harding
        30: ("1923", "1929"),  # Coolidge
        31: ("1929", "1933"),  # Hoover
        32: ("1933", "1945"),  # F.D. Roosevelt
        33: ("1945", "1953"),  # Truman
        34: ("1953", "1961"),  # Eisenhower
        35: ("1961", "1963"),  # Kennedy
        36: ("1963", "1969"),  # L.B. Johnson
        37: ("1969", "1974"),  # Nixon
        38: ("1974", "1977"),  # Ford
        39: ("1977", "1981"),  # Carter
        40: ("1981", "1989"),  # Reagan
        41: ("1989", "1993"),  # G.H.W. Bush
        42: ("1993", "2001"),  # Clinton
        43: ("2001", "2009"),  # G.W. Bush
        44: ("2009", "2017"),  # Obama
        45: ("2017", "2021"),  # Trump
    }
    
    test_characters, train_characters = load_character_data()
    
    # Generate all questions
    data = []
    for character in train_characters:
        # is_test = (character in test_characters)
        
        for president_num in range(1, 46):
            start_year = int(presidential_terms[president_num][0])
            end_year = int(presidential_terms[president_num][1])
            for i, year in enumerate(range(start_year, end_year + 1), 1):
                year_q = f"What is the {i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} year of the {president_num}th president of the United States?"
                year_a = str(year)
                data.append({"character": character['role'], "death_year": character['year'], "type": character['type'], "source": character['source'], "question": year_q, "year": year_a, "is_test": False})
    
    for character in test_characters:
        # is_test = (character in test_characters)
        
        for president_num in range(1, 46):
            start_year = int(presidential_terms[president_num][0])
            end_year = int(presidential_terms[president_num][1])
            for i, year in enumerate(range(start_year, end_year + 1), 1):
                year_q = f"What is the {i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} year of the {president_num}th president of the United States?"
                year_a = str(year)
                data.append({"character": character['role'], "death_year": character['year'], "type": character['type'], "source": character['source'], "question": year_q, "year": year_a, "is_test": True})
    
    # Create and save the DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Created CSV file: {output_file}")
    return df

def create_questions_csv(output_file="data/entity_datasets/presidential_baseline_prompt.csv", 
                        test_size=50, 
                        seed=42):
    """
    Create a CSV file with presidential questions, their year answers, and test flags.
    
    Args:
        output_file: Path to output CSV file
        test_size: Number of questions to select for test set
        seed: Random seed for reproducibility
    """
    # Presidential term data (start year, end year)
    presidential_terms = {
        1: ("1789", "1797"),  # Washington
        2: ("1797", "1801"),  # Adams
        3: ("1801", "1809"),  # Jefferson
        4: ("1809", "1817"),  # Madison
        5: ("1817", "1825"),  # Monroe
        6: ("1825", "1829"),  # J.Q. Adams
        7: ("1829", "1837"),  # Jackson
        8: ("1837", "1841"),  # Van Buren
        9: ("1841", "1841"),  # W.H. Harrison
        10: ("1841", "1845"),  # Tyler
        11: ("1845", "1849"),  # Polk
        12: ("1849", "1850"),  # Taylor
        13: ("1850", "1853"),  # Fillmore
        14: ("1853", "1857"),  # Pierce
        15: ("1857", "1861"),  # Buchanan
        16: ("1861", "1865"),  # Lincoln
        17: ("1865", "1869"),  # A. Johnson
        18: ("1869", "1877"),  # Grant
        19: ("1877", "1881"),  # Hayes
        20: ("1881", "1881"),  # Garfield
        21: ("1881", "1885"),  # Arthur
        22: ("1885", "1889"),  # Cleveland (1st)
        23: ("1889", "1893"),  # B. Harrison
        24: ("1893", "1897"),  # Cleveland (2nd)
        25: ("1897", "1901"),  # McKinley
        26: ("1901", "1909"),  # T. Roosevelt
        27: ("1909", "1913"),  # Taft
        28: ("1913", "1921"),  # Wilson
        29: ("1921", "1923"),  # Harding
        30: ("1923", "1929"),  # Coolidge
        31: ("1929", "1933"),  # Hoover
        32: ("1933", "1945"),  # F.D. Roosevelt
        33: ("1945", "1953"),  # Truman
        34: ("1953", "1961"),  # Eisenhower
        35: ("1961", "1963"),  # Kennedy
        36: ("1963", "1969"),  # L.B. Johnson
        37: ("1969", "1974"),  # Nixon
        38: ("1974", "1977"),  # Ford
        39: ("1977", "1981"),  # Carter
        40: ("1981", "1989"),  # Reagan
        41: ("1989", "1993"),  # G.H.W. Bush
        42: ("1993", "2001"),  # Clinton
        43: ("2001", "2009"),  # G.W. Bush
        44: ("2009", "2017"),  # Obama
        45: ("2017", "2021"),  # Trump
    }
    
    # Get all questions
    all_questions = make_questions()
    data = []
    
    # Process each question to extract the year answer
    for president_num in range(1, 46):
        start_year = int(presidential_terms[president_num][0])
        end_year = int(presidential_terms[president_num][1])
        for i, year in enumerate(range(start_year, end_year + 1), 1):
            prefix = "Please think step by step and carefully consider the context and only output the answer to the questions. "
            year_q = f"What is the {i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} year of the {president_num}th president of the United States?"
            year_a = str(year)
            data.append({"question": prefix + year_q, "answer": year_a, "is_test": False})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Randomly select test questions
    random.seed(seed)
    test_indices = random.sample(range(len(df)), test_size)
    df.loc[test_indices, 'is_test'] = True
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Created CSV file: {output_file}")
    print(f"Total questions: {len(df)}")
    print(f"Test questions: {df['is_test'].sum()}")
    
    return df

# Generate the CSV
# create_presidential_questions_csv()
# create_questions_csv()