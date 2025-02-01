import pandas as pd
import json

def preprocess_data(file_path):
    """Load and preprocess structured and unstructured text data."""
    df = pd.read_csv(file_path)
    df['text'] = df['text'].str.lower().str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    return df

if __name__ == '__main__':
    dataset_path = '../data/medical_data.csv'
    processed_data = preprocess_data(dataset_path)
    processed_data.to_csv('../data/processed_medical_data.csv', index=False)
    print("Data preprocessing completed.")
