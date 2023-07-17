import argparse
import os
import pandas as pd 
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from sklearn.model_selection import train_test_split


STAGE = "STAGE -02 SPLITING DATA " ## <<< change stage name 

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
    raw_data_dir = os.path.join(artifact["ARTIFACTS_DIR"], artifact["RAW_DATA_DIR"])
    raw_file_dir = os.path.join(raw_data_dir, artifact["RAW_DATA_FILE"])
    df = pd.read_csv(raw_file_dir)
    params_artifacts = params["hyper_parameters"]
    split_ratio = params_artifacts["split_ratio"]
    random_state = params_artifacts["random_state"]
    split_data_dir = os.path.join(artifact["ARTIFACTS_DIR"], artifact["SPLIT_DATA_DIR"])
    create_directories([split_data_dir])
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train_file_dir = os.path.join(split_data_dir, artifact["TRAIN_DATA"])
    test_file_dir = os.path.join(split_data_dir, artifact["TEST_DATA"])
    train.to_csv(train_file_dir, index=False)
    test.to_csv(test_file_dir, index=False)
    logging.info(f">>>>>>>>>>>Data split successfully at location {split_data_dir}<<<<<<<<")
    
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