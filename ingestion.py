import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]  # sourcedata
# output_folder_path = config["prod_deployment_path"]  # production_deployment
output_folder_path = config["output_folder_path"] # ingesteddata

directory_files = os.listdir(os.getcwd() + input_folder_path)
filenames = [filename for filename in directory_files if filename.endswith(".csv")]

#############Function for data ingestion
def merge_multiple_dataframe():

    final_data = pd.DataFrame(
        columns=[
            "corporation",
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
            "exited",
        ]
    )
    for file in filenames:
        data_df = pd.read_csv(os.getcwd() + input_folder_path + file)
        final_data = final_data.append(data_df).reset_index(drop=True)
    final_data = final_data.drop_duplicates()
    path_out = os.getcwd()+output_folder_path
    file_name = "finaldata.csv"
    final_data.to_csv(os.path.join(path_out,file_name), index=False)

    date = datetime.now()
    timenow = str(date.year)+ '/'+str(date.month)+ '/'+str(date.day)
    # record the location, name, length, and time of the ingested date file
    allrecords = [output_folder_path, filenames, len(final_data), timenow]
    with open(os.path.join(path_out,'ingestedfiles.txt'), 'w') as f:
        f.write(str(allrecords))




if __name__ == "__main__":
    merge_multiple_dataframe()
