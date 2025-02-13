import os
import csv

def summarize(path: str):
    dataset = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)        
        for row in rows:
            dataset.append(row)
    
    data_count = len(dataset)-1

    if "rag" in path:
        ir_time_idx = dataset[0].index("ir_time")
        generate_time_idx = dataset[0].index("generate_time")

        ir_time_sum = sum([float(row[ir_time_idx]) for row in dataset[1:]])
        generate_time_sum = sum([float(row[generate_time_idx]) for row in dataset[1:]])
        
        time_result = {
            "ir_time": ir_time_sum, 
            "generate_time": generate_time_sum
        }

    elif "kvcache_noCAG" in path:
        load_knowledge_time_idx = dataset[0].index("load_knowledge_time")
        generate_time_idx = dataset[0].index("generate_time")
        
        load_knowledge_time_sum = sum([float(row[load_knowledge_time_idx]) for row in dataset[1:]])
        generate_time_sum = sum([float(row[generate_time_idx]) for row in dataset[1:]])
        
        time_result = {
            "load_knowledge_time": load_knowledge_time_sum, 
            "generate_time": generate_time_sum
        }
    
    elif "kvcache_withCAG" in path:
        reset_cache_time_idx = dataset[0].index("reset_cache_time")
        generate_time_idx = dataset[0].index("generate_time")
        
        reset_cache_time_sum = sum([float(row[reset_cache_time_idx]) for row in dataset[1:]])
        generate_time_sum = sum([float(row[generate_time_idx]) for row in dataset[1:]])
        
        time_result = {
            "reset_cache_time": reset_cache_time_sum, 
            "generate_time": generate_time_sum
        }

    bert_score_idx = dataset[0].index("bert_score")
    bert_sum = sum([float(row[bert_score_idx]) for row in dataset[1:]])

    rouge1_score_idx = dataset[0].index("rouge1_score")
    rouge1_sum = sum([float(row[rouge1_score_idx]) for row in dataset[1:]])
    
    rougeL_score_idx = dataset[0].index("rougeL_score")
    rougeL_sum = sum([float(row[rougeL_score_idx]) for row in dataset[1:]])

    return data_count, {
        **time_result,
        "bert_score": bert_sum,
        "rouge1_score": rouge1_sum,
        "rougeL_score": rougeL_sum
    }
        
if __name__ == "__main__":
    dir_path = "results/new_results/"
        
    path_list = [ path for path in os.listdir(dir_path) if path.endswith(".csv") ]
    path_list = [dir_path + path for path in path_list]
    
    datasets = ["hotpotqa-train", "squad-train"]
    sizes = ["small", "medium", "large"]
    methods = [
        "rag_openai_1", 
        "rag_openai_3", 
        "rag_openai_5", 
        "rag_openai_10", 
        "rag_bm25_1", 
        "rag_bm25_3", 
        "rag_bm25_5", 
        "rag_bm25_10", 
        "kvcache_withCAG", 
        "kvcache_noCAG"
    ]
    
    for dataset in datasets:
        for size in sizes:
            target_path = f"./results_summary/{dataset}_{size}_summary.csv"
            
            for method in methods:
                topic = f"{dir_path}{dataset}_{size}_{method}"
                
                chosen_path = [path for path in path_list if path.startswith(topic+"_")]
                
                # for path in chosen_path:    
                #     count, results = summarize(path)
                #     if count != int(path.split("_")[-3]):
                #         print(f"[NotMatch {count} != {int(path.split('_')[-3])}]: {path}")
                
                total_count = 0
                total_results = {}
                
                for path in chosen_path:
                    count, results = summarize(path)
                    total_count += count
                    for key in results:
                        if key not in total_results:
                            total_results[key] = results[key]
                        else:
                            total_results[key] += results[key]
                
                with open(target_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    writer.writerow(["method", "qa_pairs"] + [key for key in total_results.keys()])
                    writer.writerow([method, total_count] + [total_results[key] / total_count for key in total_results.keys()])
                    
            print(f"{dataset}_{size} is done!")