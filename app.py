from flask import Flask, session, jsonify, request

import diagnostics
import json
import os
from scoring import score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

@app.route('/', methods=['GET'])
def root():
    return "Hello World"

#######################Prediction Endpoint
@app.route("/prediction", methods=["GET", "OPTIONS"])
def predict():
    data_name = request.args.get('filename')
    pred_lst = diagnostics.model_predictions(data_name)
    print(f'This is the predicted value list:\n {pred_lst}')
    return jsonify({'prediction':str(pred_lst)})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """check the score of the deployed model"""
    data_name = request.args.get('filename')
    score = score_model(data_name)
    return jsonify({'f1score':score})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def sumstats():
    """check means, medians, and modes for each column
     return a list of all calculated summary statistics"""
    data_name = request.args.get('filename')
    stats = diagnostics.summary_statistics(data_name)
    return jsonify({'mean_median_std': stats})

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():
    """check timing and percent NA values"""

    timing = diagnostics.execution_time()
    missing_values = diagnostics.missing_data()
    # outdated_packages = diagnostics.outdated_packages_list()
    return jsonify({'timing_second': timing,
                    'missing_values': missing_values,
                    })


if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
