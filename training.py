import pandas as pd
import pickle
import os
import logging
from sklearn.linear_model import LogisticRegression
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Training')

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

# training data
label = 'exited'
train_data = pd.read_csv(os.getcwd()+dataset_csv_path+"finaldata.csv")

# train, test = train_test_split(df_data, test_size=0.2)

y = train_data[label]
X = train_data.drop(["corporation",label], axis=1)
#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    #fit the logistic regression to your data
    model.fit(X,y)
    logger.info(f'model score: {model.score(X,y)}')
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logger.info(f'model saved location: {os.getcwd() + model_path}')
    pickle.dump(model, open(os.getcwd()+model_path+"trainedmodel.pkl", 'wb'))

if __name__=='__main__':
    train_model()