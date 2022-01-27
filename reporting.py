import pickle
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Reporting')

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
deployed_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)




##############Function for reporting
def confusion_matrix_generation():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    path_data = os.getcwd()+test_data_path+"testdata.csv"

    with open(path_data,'r') as f:
        data_test = pd.read_csv(f)
    logger.info(f"Loaded the test dataset located in: {path_data}")


    path_model = os.getcwd() + deployed_path + "trainedmodel.pkl"
    with open(path_model, 'rb') as f:
        model= pickle.load(f)
    logger.info(f"Loaded the model located in: {path_model}")

    y_test = data_test['exited']
    X_test = data_test.drop(['corporation','exited'], axis=1)
    # ypred = model.predict(X_test)
    if model_path=="/models/":
        path_confusion = os.getcwd()+ model_path+"confusion_matrix2.png"
    else:
        path_confusion = os.getcwd() + model_path + "confusion_matrix.png"

    logger.info(f"The generated confusion matrix is saved in {path_confusion}")

    plot_confusion_matrix(model, X_test, y_test)
    plt.title("Confusion matrix for Logistic Regression")
    plt.savefig(path_confusion)

if __name__ == '__main__':
    confusion_matrix_generation()
