import os
import csv
import cag.similarity as cagsim

def evaluate(path: str):
    dataset = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)        
        for row in rows:
            dataset.append(row)
    
    response_idx = dataset[0].index("generated_response")
    ground_truth_idx = dataset[0].index("ground_truth")
    
    score_idx = {
        "bert": -1,
        "rouge1": -1,
        "rougeL": -1
    }
    
    # # remove previous score
    # if "bert_score" in dataset[0]:
    #     dataset[0].remove("bert_score")
    # if "rouge1_score" in dataset[0]:
    #     dataset[0].remove("rouge1_score")
    # if "rougeL_score" in dataset[0]:
    #     dataset[0].remove("rougeL_score")
    
    if "bert_score" not in dataset[0]:
        dataset[0].append("bert_score")
        score_idx["bert"] = dataset[0].index("bert_score")
    
    if "rouge1_score" not in dataset[0]:
        dataset[0].append("rouge1_score")
        score_idx["rouge1"] = dataset[0].index("rouge1_score")
    
    if "rougeL_score" not in dataset[0]:
        dataset[0].append("rougeL_score")
        score_idx["rougeL"] = dataset[0].index("rougeL_score")
    
    for i in range(1, len(dataset)):
        response = dataset[i][response_idx]
        ground_truth = dataset[i][ground_truth_idx]
        
        # Initialize score values for each score type in one line
        dataset[i] += [0] * 3
        
        if score_idx["bert"] != -1:
            dataset[i][score_idx["bert"]] = cagsim.bert(response, ground_truth)
        if score_idx["rouge1"] != -1:
            dataset[i][score_idx["rouge1"]] = cagsim.rouge1(response, ground_truth)
        if score_idx["rougeL"] != -1:
            dataset[i][score_idx["rougeL"]] = cagsim.rougeL(response, ground_truth)

        columns_num = len(dataset[0])
        # Remove unnecessary columns
        dataset[i] = dataset[i][:columns_num]
        
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