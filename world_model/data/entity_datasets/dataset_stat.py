import csv
import pandas as pd


def get_dataset_stat(dataset_path):
    # Read the CSV file using pandas
    df = pd.read_csv(dataset_path)
    
    # Count training and testing samples
    test_count = df['is_test'].sum()  # Sum of True values (True=1, False=0)
    train_count = len(df) - test_count
    
    return train_count, test_count


if __name__ == '__main__':
    dataset_path = 'data/entity_datasets/art.csv'
    train_count, test_count = get_dataset_stat(dataset_path)
    print(f'train_count: {train_count}, test_count: {test_count}')

# The code below is no longer needed as the functionality is included in get_dataset_stat
# If you want to specifically count True values in is_test column:
def count_test_samples(dataset_path):
    df = pd.read_csv(dataset_path)
    test_count = df['is_test'].sum()
    return test_count
