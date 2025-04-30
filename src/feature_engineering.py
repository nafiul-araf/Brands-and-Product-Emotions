import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    "Load the CSV file"
    try:
        df = pd.read_csv(file_path)
        logger.debug('File loaded successfully')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the file {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file {e}')
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    "Apply the feature transformation to the data"
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['tweet_text'].values
        y_train = train_data['target'].values
        X_test = test_data['tweet_text'].values
        y_test = test_data['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test
        logger.debug('tfidf applied and data transformed')

        return train_df, test_df
    
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file {e}')
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    "Save the dataframe to a CSV file."
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f'Data saved to {file_path}')
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file {e}')
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_preprocessed.csv')
        test_data = load_data('./data/processed/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features=200)
        save_data(train_df, os.path.join("./data", "feature_trans", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "feature_trans", "test_tfidf.csv"))
        logger.debug('Feature transformed successfully')
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()