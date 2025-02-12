import os
import csv

def summarize(path: str):
    dataset = []
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)        
        for row in rows:
            dataset.append(row)

    if "rag" in path:
        ir_time_idx = dataset[0].index("ir_time")
        generate_time_idx = dataset[0].index("generate_time")
        bert_score_idx = dataset[0].index("bert_score")

        ir_time_sum = sum([float(row[ir_time_idx]) for row in dataset[1:]])
        generate_time_sum = sum([float(row[generate_time_idx]) for row in dataset[1:]])
        bert_sum = sum([float(row[bert_score_idx]) for row in dataset[1:]])
        
        return len(dataset)-1, {
            "ir_time_sum": ir_time_sum, 
            "generate_time_sum": generate_time_sum, 
            "bert_sum": bert_sum
        }

    elif "kvcache_noCAG" in path:
        load_knowledge_time_idx = dataset[0].index("load_knowledge_time")
        generate_time_idx = dataset[0].index("generate_time")
        bert_score_idx = dataset[0].index("bert_score")
        
        load_knowledge_time_sum = sum([float(row[load_knowledge_time_idx]) for row in dataset[1:]])
        generate_time_sum = sum([float(row[generate_time_idx]) for row in dataset[1:]])
        bert_sum = sum([float(row[bert_score_idx]) for row in dataset[1:]])
        
        return len(dataset)-1, {
            "load_knowledge_time_sum": load_knowledge_time_sum, 
            "generate_time_sum": generate_time_sum, 
            "bert_sum": bert_sum
        }
    
    elif "kvcache_withCAG" in path:
        reset_cache_time_idx = dataset[0].index("reset_cache_time")
        generate_time_idx = dataset[0].index("generate_time")
        bert_score_idx = dataset[0].index("bert_score")
        
        reset_cache_time_sum = sum([float(row[reset_cache_time_idx]) for row in dataset[1:]])
        generate_time_sum = sum([float(row[generate_time_idx]) for row in dataset[1:]])
        bert_sum = sum([float(row[bert_score_idx]) for row in dataset[1:]])
        
        return len(dataset)-1, {
            "reset_cache_time_sum": reset_cache_time_sum, 
            "generate_time_sum": generate_time_sum, 
            "bert_sum": bert_sum
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
                
                chosen_path = [path for path in path_list if path.startswith(topic)]
                
                # for path in chosen_path:    
                #     count, results = summarize(path)
                #     if count != int(path.split("_")[-3]):
                #         print(f"[NotMatch {count} != {int(path.split('_')[-3])}]: {path}")
                
                total_count = 0
                
                if "rag" in method:
                    ir_time_sum = 0
                    generate_time_sum = 0
                    bert_sum = 0
                    
                    for path in chosen_path:
                        count, results = summarize(path)
                        total_count += count
                        ir_time_sum += results["ir_time_sum"]
                        generate_time_sum += results["generate_time_sum"]
                        bert_sum += results["bert_sum"]
                    
                    # write to csv file
                    with open(target_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        writer.writerow(["method", "qa_pairs", "IR_time", "generate_time", "bert_score"])
                        writer.writerow([method, total_count, ir_time_sum / total_count, generate_time_sum / total_count, bert_sum / total_count])
                    
                    # with open(target_path, "a") as f:
                    #     f.write(f"{method}, qa_pairs: {total_count}, IR_time: {ir_time_sum / total_count}, generate_time: {generate_time_sum / total_count}, bert_score: {bert_sum / total_count}\n")

                elif "kvcache_noCAG" in method:
                    load_knowledge_time_sum = 0
                    generate_time_sum = 0
                    bert_sum = 0
                    
                    for path in chosen_path:
                        count, results = summarize(path)
                        total_count += count
                        load_knowledge_time_sum += results["load_knowledge_time_sum"]
                        generate_time_sum += results["generate_time_sum"]
                        bert_sum += results["bert_sum"]
                    
                    # write to csv file
                    with open(target_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        writer.writerow(["method", "qa_pairs", "load_knowledge_time", "generate_time", "bert_score"])
                        writer.writerow([method, total_count, load_knowledge_time_sum / total_count, generate_time_sum / total_count, bert_sum / total_count])
                
                elif "kvcache_withCAG" in method:
                    reset_cache_time_sum = 0
                    generate_time_sum = 0
                    bert_sum = 0
                    
                    for path in chosen_path:
                        count, results = summarize(path)
                        total_count += count
                        reset_cache_time_sum += results["reset_cache_time_sum"]
                        generate_time_sum += results["generate_time_sum"]
                        bert_sum += results["bert_sum"]
                    
                    # write to csv file
                    with open(target_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        writer.writerow(["method", "qa_pairs", "reset_cache_time", "generate_time", "bert_score"])
                        writer.writerow([method, total_count, reset_cache_time_sum / total_count, generate_time_sum / total_count, bert_sum / total_count])

            print(f"{dataset}_{size} is done!")