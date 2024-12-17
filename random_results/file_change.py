import os

def rename_files_in_order(dir_path, k):
    directory = dir_path + f"{k}/"
    for prefix in [f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_kvcache_nokv.txt", 
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_kvcache.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top1.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top3.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top5.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top10.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top20.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top1.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top3.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top5.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top10.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top20.txt",
                   ]:
        files = [f for f in os.listdir(directory) if f.startswith(prefix)]
        
        new_files = []
        for idx, file in enumerate(files, start=1):
            new_file = directory + prefix + f"_{idx}_temp"
            new_files.append(new_file)
            os.rename(directory + file, new_file)        
        
        for idx, file in enumerate(new_files, start=1):
            new_file = file[:-5]
            os.rename(file, new_file)    
            
            

def rename_files_in_order(dir_path, k):
    directory = dir_path + f"{k}/"
    for prefix in [f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_kvcache_nokv.txt", 
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_kvcache.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top1.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top3.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top5.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top10.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_bm25_top20.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top1.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top3.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top5.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top10.txt",
                   f"result_3.1-8B_k{k}_q500_hotpotqa-train_bertscore_rag_Index_openai_top20.txt",
                   ]:
        
        files = [f for f in os.listdir(directory) if f.startswith(prefix)]
        
        for idx, file in enumerate(files, start=1):
            path = directory + file
            with open(path, "r") as file:
                lines = file.readlines()
        
            updated_lines = [
                f"Result for { path }\n" if line.startswith("Result for") else line
                for line in lines
            ]
            
            # Write the updated content back to the file
            with open(path, "w") as file:
                file.writelines(updated_lines)
        
        
dir_path = "hotpotqa-train/"
for k in [16, 24, 32, 48, 64, 80]:
    rename_files_in_order(dir_path, k)

dir_path = "hotpotqa-train/"
for k in [16, 24, 32, 48, 64, 80]:
    rename_files_in_order(dir_path, k)