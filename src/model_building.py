import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import GradientBoostingClassifier
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
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

def load_data(file_path: str) -> pd.DataFrame:
    "Load the CSV file"
    try:
        df = pd.read_csv(file_path)
        logger.debug('File loaded successfully')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the file {e}')
        raise

    except FileNotFoundError as e:
        logger.error(f'File not found {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while loading the file {e}')
        raise

def train_model(train_x: np.ndarray, train_y: np.ndarray, params: dict) -> GradientBoostingClassifier:
    "Train the model"
    try:
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("The number of samples in X_train and y_train should be same")
        logger.debug(f'Initializing the model with parameters: {params}')
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug(f'Model training start with {train_x.shape[0]} samples')
        clf.fit(train_x, train_y)

        logger.debug('Model training completed')

        return clf
    except ValueError as e:
        logger.error(f'Value error during training {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while training {e}')
        raise

def save_model(model, file_path: str) -> None:
    "Save the trained model"

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f'Model saved to {file_path}')
    except FileNotFoundError as e:
        logger.debug(f'Model path not found {e}')
    except Exception as e:
        logger.error(f'Unexpected error occured while saving the model {e}')
        raise

def main():
    try:
        params = load_params('params.yaml')
        params = params['model_building']
        train_data = load_data('./data/feature_trans/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)

        model_path = 'model/model.pkl'
        save_model(clf, model_path)
    except Exception as e:
        logger.error(f'Failed to train the model {e}')
        raise

if __name__ == '__main__':
    main()