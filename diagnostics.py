import pickle

import pandas as pd
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deployed_path = os.path.join(config['prod_deployment_path'])
logs_path = os.path.join(config['logs_folder_path'])
##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    deployed_model_path = os.getcwd()+deployed_path+"trainedmodel.pkl"
    data_path = os.getcwd()+test_data_path+"testdata.csv"
    print(f'Load the trained model {deployed_model_path}')
    with open(deployed_model_path,'rb') as f:
        model = pickle.load(f)

    print(f'Load the test data {data_path}')
    with open(data_path, 'r') as f:
        test_data = pd.read_csv(f)

    y = test_data['exited']
    X = test_data.drop(['corporation','exited'], axis=1)
    prediction = model.predict(X)
    print(f"The prediction for the 5 test values are: {list(prediction)}")
    return list(prediction)

######################Function to get missing data
def missing_data():

    data_path = os.getcwd() + dataset_csv_path + 'finaldata.csv'
    with open(data_path, 'r') as f:
        data = pd.read_csv(f)
    nas = list(data.isna().sum())
    napercent =[nas[i]/len(data.index) for i in range(len(nas))]
    print(f"The percentage of NA values for each colums is given by: {napercent}")
    return napercent

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    lst_stat=[]
    features = ["lastmonth_activity", "lastyear_activity","number_of_employees"]
    data_path = os.getcwd()+dataset_csv_path+'finaldata.csv'
    with open(data_path,'r') as f:
        data = pd.read_csv(f)

    lst_stat.append(data[features].mean().to_list())
    lst_stat.append(data[features].median().to_list())
    lst_stat.append(data[features].std().to_list())
    print(f'Diagnostics: dataframe summary statistics: {lst_stat}')
    return lst_stat


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time= timeit.default_timer()
    os.system('python3 training.py')
    timing_training = timeit.default_timer() - start_time
    print(f'The time for training is: {timing_training}')

    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_ingestion = timeit.default_timer() - start_time
    print(f'The time for ingestion is: {timing_ingestion}')


    return [timing_training, timing_ingestion]

##################Function to check dependencies
def outdated_packages_list():
    # list of outdated packages
    logs = os.getcwd()+logs_path
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    with open(os.path.join(logs,"outdated.txt"), 'wb') as f:
        f.write(outdated)
    print(f"The outdated packages are listed on the third column\n {outdated.decode('utf-8')}")
    return outdated


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
