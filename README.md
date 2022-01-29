# Dynamic-Risk-Assessment-System
## Project description
In this project we want to create, deploy, and monitor a risk assessment ML model that will estimate the 
attrition risk of each of some company's clients. If the model we create and deploy is accurate, 
it will enable the client managers to contact the clients with the highest risk and avoid losing clients 
and revenue. Because the industry is dynamic and constantly changing,  and the model that has been created 
a month ago might not still be accurate today, it is necessary to setup a process and scripts to re-train, 
re-deploy, monitor and report the ML model, so the comapny get risk assessement that are as accurate as 
possible to avoid client attrition. 
nothin
<p align="center">
<img  src="./images/dynamic-risk.png"/>
</p>

### Overview of the pipeline code organisation 
```
├── Makefile
├── README.md
├── apicalls.py
├── app.py
├── config.json
├── deployment.py
├── diagnostics.py
├── environment.yml
├── fullprocess.py
├── fullprocess.sh
├── images
│   └── dynamic-risk.png
├── ingesteddata
│   ├── finaldata.csv
│   └── ingestedfiles.txt
├── ingestion.py
├── logs
│   ├── latestscore.txt
│   └── outdated.txt
├── models
│   ├── confusion_matrix.png
│   └── trainedmodel.pkl
├── practicedata
│   ├── dataset1.csv
│   └── dataset2.csv
├── practicemodels
│   ├── apireturn.txt
│   ├── confusion_matrix.png
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── production_deployment
│   ├── finaldata.csv
│   ├── ingestedfiles.txt
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── reporting.py
├── requirements.txt
├── scoring.py
├── sourcedata
│   ├── dataset3.csv
│   └── dataset4.csv
├── testdata
│   └── testdata.csv
├── training.py
└── wsgi.py
```
### Directories 
* ``practicedata``: directory which contains data used for test and practice purposes.
* ``sourcedata``: directory which contains data that needed to load and train risk models.
* ``ingesteddata``: directory which contains the compiled datasets that the ingestion script has processed.
* ``testdata``: directory contains data used for testing and evaluating models.
* ``models``: directory which contains ML models ready for production.
* ``practicemodels``: directory which contains ML models that you created for test and practice purposes.
* ``production_deployment``: directory which contains final and deployed models.
* ``logs``: directory which contains logs of packages to update
### Script files
The following are the Python files that are in the starter files:


`training.py`, a Python script meant to train an ML model

`scoring.py`, a Python script meant to score an ML model

`deployment.py`, a Python script meant to deploy a trained ML model

`ingestion.py`, a Python script meant to ingest new data

`diagnostics.py`, a Python script meant to measure model and data diagnostics

`reporting.py`, a Python script meant to generate reports about model metrics

`app.py`, a Python script meant to contain API endpoints

`wsgi.py`, a Python script to help with API deployment

`apicalls.py`, a Python script meant to call your API endpoints

`fullprocess.py`, a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed


### Config file 
This file contains five entries:

* `input_folder_path`, which specifies the location where your project will look for input data, to ingest, and to use in model training. If you change the value of input_folder_path, your project will use a different directory as its data source, and that could change the outcome of every step.
* `output_folder_path`, which specifies the location to store output files related to data ingestion. In the starter version of config.json, this is equal to /ingesteddata/, which is the directory where you'll save your ingested data later in this step.
* `test_data_path`, which specifies the location of the test dataset
* `output_model_path`, which specifies the location to store the trained models and scores.
* `prod_deployment_path`, which specifies the location to store the models in production.

### Dataset 
The column features of the data are given as follows:

* `corporation`, which contains four-character abbreviations for names of corporations
* `lastmonth_activity`, which contains the level of activity associated with each corporation over the previous month*
* `lastyear_activity`, which contains the level of activity associated with each corporation over the previous year
* `number_of_employees`, which contains the number of employees who work for the corporation
* `exited`, which contains a record of whether the corporation exited their contract (1 indicates that the corporation exited, and 0 indicates that the corporation did not exit)
* The dataset's final column, "exited", is the target variable for our predictions. The first column, "corporation", will not be used in modeling. The other three numeric columns will all be used as predictors in your ML model.

## 1. Data ingestion
```bash
> python ingestion.py
```
This script checks if there is any available data in the directory specified in `input_folder_path` and then 
saves the compiled dataset `finaldata.csv` in the directory specified in `output_model_path`. 
A record of the previous directory, names of ingested files, their length and date of ingestion is saved in a text file `ingestedfiles.txt`
that contains the names of the newly ingested data files together with 

## 2. Training, Scoring, and Deploying an ML Model
In this step we will present three scripts. One script will be for training a Logistic Regression model to 
predict attrition risk of customers, another will be for generating scoring metrics for the model, 
and the third will be for deploying the trained model.

### Model training 
```bash 
> python training.py
```
This script trains a logistic regression model from the previously ingested file to predict attrition risk and saves the model ``trainedmodel.pkl`` to the directory in `output_model_path` 

### Model Scoring
```bash
> python scoring.py
```
Computes the F1 Score of the trained model with the test dataset and then save the score `latestscore.txt` in the directory specified by `output_model_path`.

### Model Deployment 
```bash 
> python deployment.py
```
The script make copies of the trained model `trainedmodel.pkl`, score `latestscore.txt` , and a record of the ingested data `ingestedfiles.txt`, from their locations to the deployment directory `production_deployment`.

## 3. Data Diagnostics
```bash
> python diagnostics.py
```
This script contains function to do prediction, some statistics on the data, percentage of missing data, a function that times the training and the ingestion python scripts. Finally a function that displays the outdated packages.

### Model prediction 
read the deployed model and a test dataset, calculate predictions
### Summary statistics
Returns mean, median and standard deviation of the data
### Missing data
Returns the the percentage of `NA values` for each columns in the final dataset 'finaldata.csv'
### Execution time 
Returns a list containing the execution time of both ingestion script and the training python script in seconds
### Dependencies 
Returns a list of the packages that needs to be updated 
```bash
Package         Version Latest Type
--------------- ------- ------ -----
click           7.1.2   8.0.3  wheel
cycler          0.10.0  0.11.0 wheel
Flask           1.1.2   2.0.2  wheel
gunicorn        20.0.4  20.1.0 wheel
itsdangerous    1.1.0   2.0.1  wheel
Jinja2          2.11.3  3.0.3  wheel
joblib          1.0.1   1.1.0  wheel
kiwisolver      1.3.1   1.3.2  wheel
MarkupSafe      1.1.1   2.0.1  wheel
matplotlib      3.3.4   3.5.1  wheel
numpy           1.20.1  1.22.1 wheel
pandas          1.2.2   1.4.0  wheel
Pillow          8.1.0   9.0.0  wheel
pip             21.1.2  21.3.1 wheel
pyparsing       2.4.7   3.0.7  wheel
python-dateutil 2.8.1   2.8.2  wheel
pytz            2021.1  2021.3 wheel
requests        2.26.0  2.27.1 wheel
scikit-learn    0.24.1  1.0.2  wheel
scipy           1.6.1   1.7.3  wheel
seaborn         0.11.1  0.11.2 wheel
setuptools      57.0.0  60.5.0 wheel
six             1.15.0  1.16.0 wheel
threadpoolctl   2.1.0   3.0.0  wheel
Werkzeug        1.0.1   2.0.2  wheel
wheel           0.36.2  0.37.1 wheel
```
## 4. Model Reporting 
```bash
> python reporting.py
```
This script generates plot related to the ML model performance with the test data. The generated confusion matrix is shown below.

![confusion](models/confusion_matrix.png)
## 5. API 
```bash 
>python app.py
```
![flask](images/flask.png)

In the previous project we have used FastAPI. This time Flask is another tool to create and deploy api. A local web service is launched when running the python script `app.py`. Four endpoints are created within the script: 
* ``Prediction endpoint``:  
 An endpoint given at `/prediction`. This endpoint takes a dataset's file location as its input, and return the outputs of the prediction function
* ``Scoring Endpoint``:  
endpoint at `/summarystats`. This endpoint  runs the scoring.py script  and return its output.
* ``Summary Statistics Endpoint``:  
endpoint at `/summarystats`. This endpoint needs to run the summary statistics function and return its outputs.
* ``Diagnostics Endpoint``:  
endpoint at `/diagnostics`. This endpoint  runs the timing, missing data, and dependency check functions  and return their outputs.
### Calling API endpoints 
```bash 
> python apicalls.py
```
This script calls each of your endpoints, combine the outputs, and write the combined outputs to a file call `apireturns.txt`.

## Process Automation 
```bash
> python fullprocess.py
```
In this step we will create scripts that automate ML model scoring and monitoring process. This step includes checking for the criteria that will require model re-training and redeployment at 10 minutes interval

![process_automation](images/process_automation.png)

### Cron Job for the full pipeline 
We want to run the `fullprocess.py` at regular interval, without manual intervention. Therefore, we need to write a crontab file that runs each 10 minutes as an example. 
* Start the cron service
```bash 
> service cron start 
 ```
* Open the workspace's crontab file  by running 
```bash
> crontab -e 
```
* In order to check the crontab file use 
```bash
> python crontab -l 
```







