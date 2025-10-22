import json
import os
import argparse

def load_json(file_path):
    """Load JSON data from the specified file path."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save JSON data to the specified file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def combine_character_data(yearly_data, president_data, output_path):
    """Combine character data from both datasets."""
    # Create a dictionary of characters from yearly data
    yearly_chars = {item["character"]: item for item in yearly_data}
    
    # Create a dictionary of characters from president data
    president_chars = {item["character"]: item for item in president_data}
    
    # Find common characters
    common_characters = []
    for item in yearly_data:
        for president_item in president_data:
            if item["character"] == president_item["character"]:
                common_characters.append(item["character"])
    breakpoint()
    print(f"Found {len(common_characters)} common characters")
    
    # Create combined data
    combined_data = []
    
    # Process all characters from yearly data
    for character, data in yearly_chars.items():
        if character in president_chars:
            # Character exists in both datasets - combine lists
            combined_item = data.copy()
            
            # Combine lists
            for list_key in ['abstain_list', 'acc_list']:
                if list_key in data and list_key in president_chars[character]:
                    combined_item[list_key] = data[list_key] + president_chars[character][list_key]
            
            # Handle death_list separately to flip 0s and 1s in president data
            if 'death_list' in data and 'death_list' in president_chars[character]:
                # Flip 0s and 1s in president data's death_list
                flipped_president_death_list = [1 if x == 0 else 0 if x == 1 else x 
                                               for x in president_chars[character]['death_list']]
                combined_item['death_list'] = data['death_list'] + flipped_president_death_list
            
            # Handle answer_list which might be in the president data but not yearly
            if 'answer_list' in president_chars[character]:
                if 'answer_list' in data:
                    combined_item['answer_list'] = data['answer_list'] + president_chars[character]['answer_list']
                else:
                    combined_item['answer_list'] = president_chars[character]['answer_list']
            
            # Add character type and death year if available
            if 'character_type' in president_chars[character]:
                combined_item['character_type'] = president_chars[character]['character_type']
            if 'death_year' in president_chars[character]:
                combined_item['death_year'] = president_chars[character]['death_year']
                
            combined_data.append(combined_item)
        else:
            # Character only in yearly data
            combined_data.append(data)
    
    # Add characters that are only in president data
    for character, data in president_chars.items():
        if character not in yearly_chars:
            combined_data.append(data)
    
    # Save combined data
    save_json(combined_data, output_path)
    print(f"Combined data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Combine character data from two evaluation files')
    parser.add_argument('--yearly', type=str, default='Persona_Understanding/evaluation/llama/normal/yearly/final_evaluation_normal_yearly.json',
                        help='Path to yearly evaluation file')
    parser.add_argument('--president', type=str, default='Persona_Understanding/evaluation/llama/normal/four_president/final_evaluation_normal_four_president.json',
                        help='Path to four president evaluation file')
    parser.add_argument('--output', type=str, default='Persona_Understanding/evaluation/llama/normal/combined/final_evaluation_combined.json',
                        help='Path to output file')
    
    args = parser.parse_args()
    
    # Load data
    yearly_data = load_json(args.yearly)
    president_data = load_json(args.president)
    
    # Combine data
    combine_character_data(yearly_data, president_data, args.output)

if __name__ == "__main__":
    main()
