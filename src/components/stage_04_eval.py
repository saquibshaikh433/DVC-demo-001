import argparse
import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, save_json
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_log_error,mean_absolute_error, r2_score
import joblib


STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def eval_matrics(actual, predicted):
    rmse = np.sqrt(mean_squared_log_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    artifact= config["artifacts"]
    test_data_dir = os.path.join(artifact["ARTIFACTS_DIR"], artifact["SPLIT_DATA_DIR"])
    test_file_dir = os.path.join(test_data_dir, artifact["TEST_DATA"])
    test_df = pd.read_csv(test_file_dir)
    model_dir = os.path.join(artifact["ARTIFACTS_DIR"], artifact["TRAINED_MODEL_DIR"])
    model_file_name = os.path.join(model_dir, artifact["MODEL_FILE_NAME"])

    target = "quality"
    X_test = test_df.drop(target, axis = 1)
    y_test = test_df[target]

    model_lr = joblib.load(model_file_name)
    pred = model_lr.predict(X_test)

    rmse, mae, r2 = eval_matrics(y_test, pred)
    score = {"rmse": rmse,
             "mae": mae,
             "r2": r2}
    
    save_json(artifact["METRICS_JSON"], score)
    logging.info("model eval succesfull and store score at {}".format(artifact["METRICS_JSON"]))
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e