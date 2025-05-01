import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(param_path: str) -> dict:
    "Load the parameters yaml path"
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters file loaded from {param_path}')
        return params
    except FileNotFoundError as e:
        logger.error(f'File not found {e}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file: {e}')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    "Load data from a CSV file"
    try:
        df = pd.read_csv(data_url, encoding='latin-1')
        logger.debug(f'Data Loaded from {data_url}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    "Preprocess the data"
    try:
        df = df[['tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product']]
        df = df.rename(columns={'is_there_an_emotion_directed_at_a_brand_or_product': 'target'})
        logger.debug('Basic preprocessing completed')
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    "Save the train and test data"
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f'Train and test data saved to {raw_data_path}')
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file: {e}')
        raise

def main():
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/nafiul-araf/Brands-and-Product-Emotions/main/experiments/emotions.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df=df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data=train_data, test_data=test_data, data_path='./data')
        logger.debug('All process done sucessfully')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()