import pickle

import training
import deployment
from diagnostics import model_predictions
from scoring import score_model
from sklearn.metrics import f1_score
import reporting
import pandas as pd
import json
import os
import ast
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])  # sourcedata
output_folder_path = os.path.join(config['output_folder_path']) # ingesteddata
test_data_path = os.path.join(config['test_data_path'])
deployed_path = os.path.join(config['prod_deployment_path']) # production_deployment
logs_path = os.path.join(config['logs_folder_path'])
##################Check and read new data

def check_new_data():
    """check if there is new data (not ingested) in sourcedata directory
        Return the data files """

    ingested_path = os.getcwd() + deployed_path + 'ingestedfiles.txt'
    with open(ingested_path, 'r') as f:
        ingested_info = ast.literal_eval(f.read())
    print(f'ingested files info: {ingested_info}')

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    sourcedata_path = os.getcwd() + input_folder_path
    ingested_files = [f for f in os.listdir(sourcedata_path) if f.endswith('.csv')]

    return set(ingested_files).difference(set(ingested_info[1]))

def check_drift():
    """Check with the newly ingested data if there is model drift
       Return True if new f1score is lower than the old one"""

    score_path = os.getcwd()+deployed_path+"latestscore.txt"
    output_data = os.getcwd()+output_folder_path
    with open(score_path, 'r') as f:
        old_f1score = float(f.read())

    # Generate the new score for the new data
    data = pd.read_csv(os.path.join(output_data,"finaldata.csv"))
    y = data['exited']
    ypred = model_predictions(data)
    new_f1score = f1_score(y, ypred)
    print(f"new f1score: {new_f1score} vs old f1score: {old_f1score}")

    return new_f1score < old_f1score

def ingest_new_data():
    """Run the ingestion python file"""
    subprocess.run(['python', 'ingestion.py'], stdout=subprocess.PIPE)

def retrain_model():
    """Retrain the model with the new data"""
    subprocess.run(['python', 'training.py'], stdout=subprocess.PIPE)


def redeploy_model():
    """copy the latest pickle file, the latestscore.txt value,
        and the ingestedfiles.txt file into the deployment directory"""
    subprocess.run(['python', 'deployment.py'], stdout=subprocess.PIPE)

def diagnostics_and_reporting():
    """Run the scoring, diagnostics and reporting process"""
    subprocess.run(['python', 'scoring.py'], stdout=subprocess.PIPE)
    subprocess.run(['python', 'diagnostics.py'], stdout=subprocess.PIPE)
    subprocess.run(['python', 'reporting.py'], stdout=subprocess.PIPE)



def main():
    if check_new_data():
        ingest_new_data()
        drift = check_drift()
        if drift:
            retrain_model()
            redeploy_model()
            diagnostics_and_reporting()






if __name__=="__main__":
    main()

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







