

import pandas as pd
import glob
import os

def read_all_csvs(directory_path):
    """
    Read all CSV files in the specified directory into a dictionary of DataFrames.
    
    Args:
        directory_path (str): Path to the directory containing CSV files
        
    Returns:
        dict: Dictionary with filenames as keys and pandas DataFrames as values
    """
    # Create an empty dictionary to store our dataframes
    csv_dict = {}
    
    # Get list of all CSV files in directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    # Read each CSV file into a DataFrame
    for csv_file in csv_files:
        try:
            # Get filename without path and extension
            filename = os.path.basename(csv_file).replace('.csv', '')
            
            # Read CSV into DataFrame
            df = pd.read_csv(csv_file)
            
            # Store in dictionary
            csv_dict[filename] = df
            
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
    
    return csv_dict

import json

hotpot_path = "../datasets/hotpotqa/hotpot_train_v1.1.json"
squad_path = "../datasets/squad/train-v1.1.json"
#Load the dataset from json
def load_hotpot():
    data = json.load(open(hotpot_path, "r"))
    qa_map = {}
    for i, qa in enumerate(data):
        print(qa.keys())
        qa_map[qa["answer"]] = qa["question"]

def load_qa():
    data = json.load(open(squad_path, "r"))
    qa_map = {}
    for i, d in enumerate(data["data"]):
        for p in d["paragraphs"]:
            for q in p["qas"]:
                qa_map[q["answers"][0]["text"]] = q["question"]
    return qa_map

qa_map = load_qa()

directory = "new_results"
dataframes = read_all_csvs(directory)
for filename, df in dataframes.items():
    if not filename.startswith("squad"):
        continue
    filename = filename.split("new_results/")[0]

    df['question'] = df['ground_truth'].map(qa_map)
    #for index, row in df.iterrows():
    #    row['question'] = qa_map[row['ground_truth']]

    df.to_csv(f'results_w_question/{filename}.csv', index=False)





