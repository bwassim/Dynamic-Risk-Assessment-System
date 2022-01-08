from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path']) 


outpath = os.getcwd()+model_path
#################Function for model scoring
def score_model():
    with open(outpath+"trainedmodel.pkl",'rb') as f:
        model  = pickle.load(f)
   # load test data
    test_data = pd.read_csv(os.getcwd()+test_data_path+"testdata.csv")
    y = test_data['exited']
    X = test_data.drop(['corporation','exited'],axis=1)
    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted,y)
    print(f'\n{f1score}')
    # write the result to the latestscore.txt
    with open(os.path.join(outpath,'latestscore.txt'), 'w' ) as f:
        f.write(str(f1score))
if __name__=="__main__":
    score_model()

