from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Deployment')

##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])
####################function for deployment
def store_model_into_pickle():
    """copy the latest pickle file, the latestscore.txt value,
    and the ingestedfiles.txt file into the deployment directory"""

    logger.info("Store pickle model and latestscore.txt into deploy directory")

    model_pickle_path = os.getcwd() + output_model_path + "trainedmodel.pkl"
    score_txt_path = os.getcwd() + output_model_path + "latestscore.txt"
    deploy_path = os.getcwd() + prod_deployment_path

    for file in [model_pickle_path, score_txt_path]:
        shutil.copy2(file, deploy_path)

    logger.info(f"Copied the newly trained model: {model_pickle_path} into {deploy_path}")
    logger.info(f"Copied the new score: {score_txt_path} into {deploy_path}")
    ingestedfile_txt_path = os.getcwd() + dataset_csv_path + "ingestedfiles.txt"

    logger.info(f"Copy {ingestedfile_txt_path} into {deploy_path}")
    shutil.copy2(ingestedfile_txt_path, deploy_path)


if __name__ == "__main__":
    store_model_into_pickle()
