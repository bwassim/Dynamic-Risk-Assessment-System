import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Scoring')

#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])
deployed_path = os.path.join(config["prod_deployment_path"])
#################Function for model scoring
def score_model(file_name=None):
    """
    - Use the test data file to generate f1score of the trained model and
    - Write the new f1score into production_deploymeny/latestscore.txt
    Return the f1score"""

    if file_name is None:
        data_path = os.getcwd() + test_data_path + "testdata.csv"
    else:
        data_path = os.getcwd() + test_data_path + file_name

    logger.info(f"Load the trained model: {os.getcwd() + model_path+'trainedmodel.pkl'} ")
    with open(os.getcwd() + model_path+"trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)

    logger.info(f"Load the test data {data_path}")
    test_data = pd.read_csv(data_path)
    y = test_data["exited"]
    X = test_data.drop(["corporation", "exited"], axis=1)
    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)

    logger.info(f"F1 score on test data is: {f1score}")

    with open(os.getcwd() + model_path + "latestscore.txt", "w") as f:
        f.write(str(f1score))

    logger.info(f"F1 score is saved in: {os.getcwd() + model_path + 'latestscore.txt'} ")

    return f1score

if __name__=="__main__":
    score_model()