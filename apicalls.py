import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

def check_app_port(port=8000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 8000))
    if result == 0:
        print("API calls - check_app_port() ... port {} is open".format(port))
    else:
        print("API calls - check_app_port() ... port {} is not open".format(port))
    sock.close()
    return result == 0

#Call each API endpoint and store the responses
def run_api_endpoints():
    print(f"API for prediction result: by default testdata/testdata.csv is chosen \n {URL+'/prediction'}")
    response1 = requests.get(URL+'/prediction').text
    print(f"API for  f1score for the test data: {URL+'/scoring'}")
    response2 = requests.get(URL+'/scoring').text
    print(f"API for summary statistics: {URL+'/summarystats'}")
    response3 = requests.get(URL+'/summarystats').text
    print(f" API for execution time, missing data {URL+'/diagnostics'}")
    response4 = requests.get(URL+'/diagnostics').text
    #combine all API responses
    responses = [response1, response2, response3, response4]
    # save to file apireturn.txt
    path = os.getcwd() +model_path
    with open(os.path.join(path, "apireturn.txt"), 'w') as f:
        f.writelines(responses)

#write the responses to your workspace

if __name__=="__main__":
    run_api_endpoints()

