import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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

def load_model(model_path: str):
    "Load the model"
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f'Model loaded from {model_path}')
        return model

    except FileNotFoundError as e:
        logger.error(f'File not found {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while loading the model {e}')
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

def evaluate_model(clf, test_x: np.ndarray, test_y: np.ndarray) -> dict:
    "Evaluate the model"
    try:
        y_pred = clf.predict(test_x)
        
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='weighted')
        recall = recall_score(test_y, y_pred, average='weighted')

        metrics_dicts = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        logger.debug('Model evaluation metrics calculated')
        return metrics_dicts
    
    except Exception as e:
        logger.error(f'Unexpected error occured while evaluating the model {e}')
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    "Save the evaluation results"

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=3)
        logger.debug(f'Metrics saved to {file_path}')

    except Exception as e:
        logger.error(f'Unexpected error occured while saving the metrics {e}')
        raise

def main():
    try:
        params = load_params('params.yaml')
        clf = load_model('./model/model.pkl')
        test_data = load_data('./data/feature_trans/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])

            live.log_params(params)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error(f'Unexpected error occured in the evaluation process {e}')
        raise

if __name__ == '__main__':
    main()