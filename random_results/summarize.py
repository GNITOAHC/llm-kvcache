import os

def main():
    dir_path = "hotpotqa-train/"
    
    for k in [16, 24, 32, 48, 64, 80]:
        collect_data(dir_path, k)
    
    for k in [16, 24, 32, 48, 64, 80]:
        summarize_file(dir_path, k)

def collect_data(dir_path, k):
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
        
        result_dir = dir_path + f"result_{k}/"
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        result_file = result_dir + f"result_{k}_" + prefix.split("bertscore_")[1]
        
        index = 0
        
        for idx, file in enumerate(files, start=1):
            with open(directory + file, "r") as f:
                lines = f.readlines()
            result = []
            prepare = []
            average = []
            info_time = []
            
            for line in lines:
                if line.startswith("Result for"):
                    result.append(line)
                elif line.startswith("Prepare time"):
                    prepare.append(line)
                elif line.startswith("Average Semantic Similarity"):
                    average.append(line)
                elif line.startswith("cache") or line.startswith("retrieve"):
                    info_time.append(line)
            
            datas = zip(result, prepare, average, info_time)
            
            with open(result_file, "a") as f:
                for data in datas:
                    index += 1
                    for d in data:
                        f.writelines( f"[{index}]" + d)
                    f.writelines("\n")
                    
def summarize_file(dir_path, k):
    directory = dir_path + f"result_{k}/"
    for filename in [f"result_{k}_kvcache_nokv.txt", 
                   f"result_{k}_kvcache.txt",
                   f"result_{k}_rag_Index_bm25_top1.txt",
                   f"result_{k}_rag_Index_bm25_top3.txt",
                   f"result_{k}_rag_Index_bm25_top5.txt",
                   f"result_{k}_rag_Index_bm25_top10.txt",
                   f"result_{k}_rag_Index_bm25_top20.txt",
                   f"result_{k}_rag_Index_openai_top1.txt",
                   f"result_{k}_rag_Index_openai_top3.txt",
                   f"result_{k}_rag_Index_openai_top5.txt",
                   f"result_{k}_rag_Index_openai_top10.txt",
                   f"result_{k}_rag_Index_openai_top20.txt",
                   ]:
        result_file = directory + filename
        
        with open (result_file, "r") as f:
            lines = f.readlines()
        
        prepare = []
        similarity = []
        info_time = []
        generate_time = []
        
        info_type = ""
        
        for line in lines:
            if line == "\n":
                continue
            line = line.split("]")[1]
            if line.startswith("Prepare time: "):
                item = line.split("Prepare time: ")[1]
                prepare.append(float(item))
            elif line.startswith("Average Semantic Similarity: "):
                item = line.split("Average Semantic Similarity: ")[1]
                similarity.append(float(item))
            elif line.startswith("cache"):
                item1 = line.split("cache time: ")[1].split(",")[0]
                item2 = line.split("generate time: ")[1].split("\n")[0]
                info_time.append(float(item1))
                generate_time.append(float(item2))
                info_type = "cache time:"
            elif line.startswith("retrieve"):
                item1 = line.split("retrieve time: ")[1].split(",")[0]
                item2 = line.split("generate time: ")[1].split("\n")[0]
                info_time.append(float(item1))
                generate_time.append(float(item2))
                info_type = "retrieve time:"
            
        avg_prepare = sum(prepare) / len(prepare)
        avg_similarity = sum(similarity) / len(similarity)
        avg_info_time = sum(info_time) / len(info_time)
        avg_generate_time = sum(generate_time) / len(generate_time)
        
        with open(dir_path + f"result_{k}_summary.txt", "a") as f:
            f.writelines(f"==> {len(prepare) * k} Questions from {filename}\n")
            f.writelines(f"Prepare time: {avg_prepare}\n")
            f.writelines(f"Average Semantic Similarity: {avg_similarity}\n")
            f.writelines(f"{info_type} {avg_info_time} \t generate Time: {avg_generate_time}\n")
            f.writelines("\n")

if __name__ == "__main__":
    main()
    pass
