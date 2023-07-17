import argparse
import os
import pandas as pd 
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random



STAGE = "Get Data Stage-01" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    data_url = config["data_source"]
    df = pd.read_csv(data_url, sep=";")
    artifact = config['artifacts']
    artifact_dir = os.path.join(artifact['ARTIFACTS_DIR'], artifact['RAW_DATA_DIR'])
    file_dir = artifact["RAW_DATA_FILE"]
    create_directories([artifact_dir])
    file_path_dir = os.path.join(artifact_dir, file_dir)
    df.to_csv(file_path_dir, sep=",", index=False)
    logging.info(f">>>>> data ingestion step run successfully at {file_path_dir}")

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