import os
import csv
import cag.similarity as cagsim

def evaluate(path: str):
    dataset = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)        
        for row in rows:
            dataset.append(row)
    
    if dataset[0][-1] == "bert_score":
        return
    
    dataset[0].append("bert_score")

    response_idx = dataset[0].index("generated_response")
    ground_truth_idx = dataset[0].index("ground_truth")

    for i in range(1, len(dataset)):
        response = dataset[i][response_idx]
        ground_truth = dataset[i][ground_truth_idx]
        dataset[i].append(cagsim.bert(response, ground_truth))
    
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataset)
    
    print("[ Done ]: ", path)

if __name__ == "__main__":
    dir_path = "results/new_results/"
        
    path_list = [ path for path in os.listdir(dir_path) if path.endswith(".csv") ]
    # print(path_list)
    path_list = [dir_path + path for path in path_list]
    
    for path in path_list:
        evaluate(path)