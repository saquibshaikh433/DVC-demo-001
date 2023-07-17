import argparse
import os
import pandas as pd 
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib


STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifact= config["artifacts"]
    train_data_dir = os.path.join(artifact["ARTIFACTS_DIR"], artifact["SPLIT_DATA_DIR"])
    train_file_dir = os.path.join(train_data_dir, artifact["TRAIN_DATA"])
    train_df = pd.read_csv(train_file_dir)
    model_dir = os.path.join(artifact["ARTIFACTS_DIR"], artifact["TRAINED_MODEL_DIR"])
    create_directories([model_dir])
    model_file_name = os.path.join(model_dir, artifact["MODEL_FILE_NAME"])

    target = "quality"
    X = train_df.drop(target, axis = 1)
    y = train_df[target]

    params_artifacts = params["model_params"]["ElasticNet"]
    alpha = params_artifacts["alpha"]
    l1_ratio = params_artifacts["l1_ratio"]
    random_state = params["hyper_parameters"]["random_state"]

    model_lr = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state
    )
    model_lr.fit(X, y)
    joblib.dump(model_lr, model_file_name)
    logging.info(f"training stage run successfully and model save at {model_file_name}")
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e