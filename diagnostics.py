import pickle

import pandas as pd
import timeit
import os
import json
import subprocess
from io import StringIO
import logging
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Diagnostics')

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deployed_path = os.path.join(config['prod_deployment_path'])
logs_path = os.path.join(config['logs_folder_path'])
##################Function to get model predictions
def model_predictions(file_name=None):
    """read the deployed model and a test dataset, calculate predictions
       if file_name is empty we use the testdata.csv to """

    deployed_model_path = os.getcwd()+deployed_path+"trainedmodel.pkl"
    if file_name is None:
        data_path = os.getcwd()+test_data_path+"testdata.csv"
        logger.info(f'Load the test data {data_path}')
        with open(data_path, 'r') as f:
            test_data = pd.read_csv(f)
    else:
        test_data = file_name

    logger.info(f'Load the trained model {deployed_model_path}')
    with open(deployed_model_path,'rb') as f:
        model = pickle.load(f)

    X = test_data.drop(['corporation','exited'], axis=1)
    prediction = model.predict(X)
    logger.info(f"The prediction for the {len(prediction)} test values are: {list(prediction)}")
    return list(prediction)

######################Function to get missing data
def missing_data():

    data_path = os.getcwd() + dataset_csv_path + 'finaldata.csv'
    with open(data_path, 'r') as f:
        data = pd.read_csv(f)
    # nas = list(data.isna().sum())
    # napercent =[nas[i]/len(data.index) for i in range(len(nas))]
    na_percent = data.isna().mean()*100
    logger.info(f"The percentage of NA values for each columns is given by \n: {na_percent}")
    return na_percent.to_list()

##################Function to get summary statistics
def summary_statistics(filename=None):
    if filename is None:
        data_path = os.getcwd()+test_data_path+"testdata.csv"
    else:
        data_path = os.getcwd() + test_data_path + filename
    #calculate summary statistics here
    lst_stat=[]
    features = ["lastmonth_activity", "lastyear_activity","number_of_employees"]

    with open(data_path,'r') as f:
        data = pd.read_csv(f)

    lst_stat.append(data[features].mean().to_list())
    lst_stat.append(data[features].median().to_list())
    lst_stat.append(data[features].std().to_list())
    logger.info(f'Diagnostics: summary statistics:[mean, median, std] \n {lst_stat}')
    return lst_stat


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time= timeit.default_timer()
    os.system('python3 training.py')
    timing_training = timeit.default_timer() - start_time
    logger.info(f'The time for training is: {timing_training}')

    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_ingestion = timeit.default_timer() - start_time
    logger.info(f'The time for ingestion is: {timing_ingestion}')


    return [timing_training, timing_ingestion]

##################Function to check dependencies
def outdated_packages_list():
    # list of outdated packages
    logs = os.getcwd()+logs_path
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    df = pd.read_csv(StringIO(outdated.decode('utf-8')), index_col='Package', sep=r"\s+", skiprows=[1], engine='python')
    print(df)
    with open(os.path.join(logs,"outdated.txt"), 'wb') as f:
        f.write(outdated)
    logger.info(f"The outdated packages are listed on the third column\n {outdated.decode('utf-8')}")
    return outdated.decode('utf-8')


if __name__ == '__main__':
    model_predictions()
    summary_statistics()
    missing_data()
    execution_time()
    outdated_packages_list()





    
