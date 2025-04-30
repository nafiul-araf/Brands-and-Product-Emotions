import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    "Transform the text by text lowering, tokenizing, removing special characters, stopwords, punctuations and stemming."
    
    # Lower case the text
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove special characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming the texts
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]

    logger.debug('Text transform completed')
    # Join the tokens back into a single string
    return ' '.join(text)

def preprocess_df(df, text_column = 'tweet_text', target_column = 'target'):
    "Preprocess the dataframe by label encoding the target column, removing the duplicates and nulls and apply the txt transformations"

    try:
        logger.debug('Preprocessing starts...')

        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove the duplicates
        df = df.drop_duplicates(keep='first')
        df = df.reset_index(drop=True)
        logger.debug('Duplicates removed')

        # Remove the nulls
        df = df.dropna()
        df = df.reset_index(drop=True)
        logger.debug('Nulls removed')

        # Transform the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error(f'Column not found {e}')
        raise
    
    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file: {e}')
        raise

def main(text_column = 'tweet_text', target_column = 'target'):
    "Main function to load raw data, preprocess it, and save the processed data."

    try:
        # Fetch the data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_preprocessed_data = preprocess_df(train_data, text_column, target_column)
        test_preprocessed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join('./data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        train_preprocessed_data.to_csv(os.path.join(data_path, 'train_preprocessed.csv'), index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

        logger.debug(f'Processed data saved to {data_path}')

    except FileNotFoundError as e:
        logger.error(f'File not found {e}')
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f'No data: {e}')
    
    except Exception as e:
        logger.error(f'Failed to complete the data transformation process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()