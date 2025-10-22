from .common import *
import random
import pandas as pd
import json

ART_PROMPTS = {
    # 'empty': '',
    # 'random': '',
    'release': 'When was the release date of ',
    # 'empty_all_caps': '',
}



def make_book_entity_df(raw_data_dir, min_wiki_page_views=5000, min_year=1900):
    book_df = pd.read_csv(os.path.join(raw_data_dir, 'books.csv'))
    book_page_view_df = pd.read_csv(
        os.path.join(raw_data_dir, 'book_page_views.csv'),
        names=['entity', 'page_views'], skiprows=1
    )

    book_df['page_name'] = book_df['wikiPage'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])
    book_df = book_df.drop(columns=['book', 'wikiPage'])
    book_df = book_df.rename(
        columns={'author': 'creator', 'pageCount': 'length', 'releaseDate': 'release_date'})
    book_df = book_df[book_df.release_date.apply(
        lambda x: x[:4]).astype(int) > min_year]
    book_df = book_df.join(book_page_view_df.set_index('entity'), on='page_name')\
        .sort_values('release_date')\
        .drop_duplicates(['title', 'creator'])\
        .sort_values('page_views', ascending=False)\
        .dropna(subset=['title', 'page_views'])

    book_df = book_df.query('page_views > @min_wiki_page_views')

    book_df = book_df[~book_df.creator.isna()]

    return book_df


def make_movie_entity_df(raw_data_dir, min_wiki_page_views=5000, min_year=1900):
    movie_df = pd.read_csv(os.path.join(raw_data_dir, 'movies.csv'))
    movie_page_view_df = pd.read_csv(
        os.path.join(raw_data_dir, 'movie_page_view.csv'),
        names=['entity', 'page_views'], skiprows=1
    )
    movie_df['page_name'] = movie_df['wikiPage'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])
    movie_df = movie_df.drop(columns=['movie', 'wikiPage'])
    movie_df = movie_df.rename(
        columns={'director': 'creator', 'runtime': 'length', 'releaseDate': 'release_date'})
    movie_df = movie_df[movie_df.release_date.apply(
        lambda x: x[:4]).astype(int) > min_year]
    movie_df = movie_df.join(movie_page_view_df.set_index('entity'), on='page_name')\
        .sort_values('release_date')\
        .drop_duplicates(['title', 'creator'])\
        .sort_values('page_views', ascending=False)\
        .dropna(subset=['title', 'page_views'])

    movie_df = movie_df.query('page_views > @min_wiki_page_views')

    movie_df = movie_df[~movie_df.creator.isna()]

    return movie_df


def make_song_entity_df(raw_data_dir, min_wiki_page_views=5000, min_year=1900):
    song_df = pd.read_csv(os.path.join(raw_data_dir, 'songs.csv'))
    song_page_view_df = pd.read_csv(
        os.path.join(raw_data_dir, 'song_page_view.csv'),
        names=['entity', 'page_views'], skiprows=1
    )
    song_df['page_name'] = song_df['wikiPage'].apply(
        lambda x: x[len('http://en.wikipedia.org/wiki/'):])
    song_df = song_df.drop(columns=['song', 'wikiPage'])
    song_df = song_df.rename(
        columns={'artist': 'creator', 'releaseDate': 'release_date'})
    song_df = song_df[song_df.release_date.apply(
        lambda x: x[:4]).astype(int) > min_year]
    song_df = song_df.join(song_page_view_df.set_index('entity'), on='page_name')\
        .sort_values('release_date')\
        .drop_duplicates(['title', 'creator'])\
        .sort_values('page_views', ascending=False)\
        .dropna(subset=['title', 'page_views'])

    song_df = song_df.query('page_views > @min_wiki_page_views')

    song_df = song_df[~song_df.creator.isna()]

    return song_df


def sanitize_title(title):
    try:
        title = title.strip()
        if title[0] == '"':
            title = title[1:]
        if title[-1] == '"':
            title = title[:-1]
        title = title.strip()
        while title[-1] == '.':
            title = title[:-1]
        title = title.strip()
        return title
    except IndexError:
        return ""


def make_art_entity_dataset(raw_data_dir, min_wiki_page_views=5000, min_year=1949, test_ratio=0.2):
    book_df = make_book_entity_df(
        raw_data_dir, min_year=min_year, min_wiki_page_views=min_wiki_page_views)
    movie_df = make_movie_entity_df(
        raw_data_dir, min_year=min_year, min_wiki_page_views=min_wiki_page_views)
    song_df = make_song_entity_df(
        raw_data_dir, min_year=min_year, min_wiki_page_views=min_wiki_page_views)

    book_df['entity_type'] = 'book'
    movie_df['entity_type'] = 'movie'
    song_df['entity_type'] = 'song'

    art_df = pd.concat([book_df, movie_df, song_df])

    art_df['title'] = art_df['title'].apply(sanitize_title)
    art_df = art_df.loc[art_df['title'].apply(lambda x: len(x)) > 1]
    art_df = art_df.reset_index(drop=True)

    unique_creators = art_df.creator.unique()
    n = len(unique_creators)
    #FIXME: instead of random, it should split
    test_creators = np.random.choice(
        unique_creators, size=int(n*test_ratio), replace=False)
    test_set = np.array([
        page_name in test_creators for page_name in art_df.creator.values
    ])

    art_df['is_test'] = test_set

    save_path = os.path.join('data', 'entity_datasets', 'art.csv')
    art_df.to_csv(save_path, index=False)


def load_character_data():
    # Path is relative to the project root
    character_path = 'data/full_question.json'
    
    with open(character_path, 'r') as f:
        return json.load(f)


def make_art_prompt_dataset(short_prompt, prompt, tokenizer, art_df, use_character=True):

    dataset_strings = []
    # real_list = ['Alexander Graham Bell', 'Amelia Earhart', 'Babe Ruth', 'Claude Debussy', 'Claude Monet', 'Edgar Degas', 'F. Scott Fitzgerald', 'Florence Nightingale', 'Mahatma Gandhi', 'Mark Twain', 'Marilyn Miller', 'Max Planck', 'Nikola Tesla', 'Oscar Wilde', 'Sergei Rachmaninoff', 'Thomas Edison', 'W. C. Fields', 'Wilbur Wright', 'Marie Curie', 'Harriet Tubman', 'George Washington Carver', 'Madam C. J. Walker', 'Sigmund Freud']
    # real_list_19801990 =['Andy Kaufman', 'Bob Marley', 'Cary Grant', 'Desi Arnaz', 'Dick Shawn', 'Enzo Ferrari', 'Grace Kelly', 'Graham Chapman', 'Henry Fonda', 'Ingrid Bergman', 'Jean-Michel Basquiat', 'John Belushi', 'Laurence Olivier', 'Lee Strasberg', 'Lucille Ball', 'Marvin Gaye', 'Orson Welles', 'Richard Feynman', 'Rita Hayworth', 'Roy Orbison']
    # death_year = [1984,1981,1986,1986,1987,1988,1982,1989,1982,1982,1988,1982,1989,1982,1989,1984,1985,1988,1987,1988]
    
    # historical_figures = [
    #     {"role": "Andy Kaufman", "year": 1984},
    #     {"role": "Bob Marley", "year": 1981},
    #     {"role": "Cary Grant", "year": 1986},
    #     {"role": "Desi Arnaz", "year": 1986},
    #     {"role": "Dick Shawn", "year": 1987},
    #     {"role": "Enzo Ferrari", "year": 1988},
    #     {"role": "Grace Kelly", "year": 1982},
    #     {"role": "Graham Chapman", "year": 1989},
    #     {"role": "Henry Fonda", "year": 1982},
    #     {"role": "Ingrid Bergman", "year": 1982},
    #     {"role": "Jean-Michel Basquiat", "year": 1988},
    #     {"role": "John Belushi", "year": 1982},
    #     {"role": "Laurence Olivier", "year": 1989},
    #     {"role": "Lee Strasberg", "year": 1982},
    #     {"role": "Lucille Ball", "year": 1989},
    #     {"role": "Marvin Gaye", "year": 1984},
    #     {"role": "Orson Welles", "year": 1985},
    #     {"role": "Richard Feynman", "year": 1988},
    #     {"role": "Rita Hayworth", "year": 1987}
    # ]
    for _, row in art_df.iterrows():
        apos = "'s" if row.creator[-1] != 's' else "'"
        prompt_suffix = f"{row.creator}{apos} {row.title}"

        if short_prompt.endswith('all_caps'):
            prompt_suffix = prompt_suffix.upper()

        dataset_strings.append(prompt + prompt_suffix)

    # Add character information for release prompt
    # if use_character and short_prompt == 'release':
    #     try:
    #         characters = load_character_data()
    #         # Only use real people (label 0)
    #         real_people = [char for char in characters if char['label'] == 0]
            
    #         # Create new strings with character names
    #         new_strings = []
    #         for s in dataset_strings:
    #             character = random.choice(real_people)
    #             new_strings.append(f"You should play the role. You are {character['name']}, {s}")
    #             print(new_strings[-1])
    #         dataset_strings = new_strings
    #     except Exception as e:
    #         print(f"Warning: Could not add characters to art prompts: {e}")
    if use_character and short_prompt == 'release':
        try:
            # Load characters from the JSON file
            character_data = load_character_data()
            
            # Extract characters from the nested structure
            real_people = []
            # fictional_chars = []
            
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
            
            # If we don't have enough characters for training, use all characters for both
            if not train_characters:
                train_characters = test_characters
            
            new_strings = []
            for i, s in enumerate(dataset_strings):
                # Check if this is a test data point
                if art_df.iloc[i].is_test:
                    # For test data, use characters from test set
                    character = random.choice(test_characters)
                else:
                    # For non-test data, use characters from training set
                    character = random.choice(train_characters)
                
                year = character.get('year', 2000)
                char_role = character['role']
                
                # Add source information for fictional characters
                character_description = char_role
                if character.get('type') == 'mix_fictional' and 'source' in character:
                    character_description = f"{char_role} from {character['source']}"
                
                new_strings.append(f" You must strictly adhere to the role assigned to you and respond as if you are that character or person. You should only possess knowledge that your role would have during their lifetime. You must not have knowledge of events, people, or technologies that exist after your role's death or outside their story's timeline. You should check the year of your death and year of the events in the questions. If the year of the event is after your death, you should abstain and not answer. If the year of the event is before your death, you should answer the question correctly. Please only output the answer to the questions. You died in {year}. This year is 2023. You are {character_description}, {s}")
                # new_strings.append(f" You should play the role given to you. Please only output the answer to the questions. You are {character_description}, {s}")
                print(new_strings[-1])
            dataset_strings = new_strings
        except Exception as e:
            print(f"Warning: Could not add characters to art prompts: {e}")
            import traceback
            traceback.print_exc()  # Print the full error trace for debugging

    token_ids = tokenizer.batch_encode_plus(
        dataset_strings,
        return_tensors='pt',
        padding=True,
        add_special_tokens=False,
        return_attention_mask=False
    )['input_ids']

    if short_prompt == 'random':
        random_prompts = torch.randint(
            low=100, high=token_ids.max().item(),
            size=(token_ids.shape[0], 10),
            dtype=torch.long
        )
        token_ids = torch.cat([random_prompts, token_ids], dim=1)

    # add bos token
    token_ids = torch.cat([
        torch.ones(token_ids.shape[0], 1,
                   dtype=torch.long) * tokenizer.bos_token_id,
        token_ids], dim=1
    )

    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
    entity_mask[:, prompt_tokens] = False
    entity_mask[token_ids == tokenizer.pad_token_id] = False

    dataset = datasets.Dataset.from_dict({
        'entity': art_df.title.values.tolist(),
        'creator': art_df.creator.values.tolist(),
        'input_ids': token_ids.tolist(),
        'entity_mask': entity_mask.tolist(),
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
            format='%Y-%m-%d'
        )
        return time.values.astype(int)

    def make_and_save_tokenized_datasets(self, tokenizer, model_family, prompt_type):
        if self.entity_data is None:
            self.entity_data = self.load_entity_data()

        for short_prompt, full_prompt in self.prompt_dict.items():
            dataset = make_art_prompt_dataset(
                full_prompt, tokenizer, self.entity_data)

            save_path = self.prompt_data_path(short_prompt, model_family, prompt_type)
            dataset.save_to_disk(save_path)
