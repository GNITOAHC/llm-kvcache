#!/bin/bash

# hotpotqa-train
# "small": 16  # 16 docs ≈ 21k tokens  
# "medium": 32 # 32 docs ≈ 43k tokens  
# "large": 64  # 64 docs ≈ 85k tokens  

indices=("bm25" "openai")
top_k=("1" "3" "5" "10")
total_qa=500

dataset="hotpotqa-train"
sizes_qa_list=(
    # "small 16"
    # "medium 32"
    "large 64"
)

for entry in "${sizes_qa_list[@]}"; do
    size=$(echo "$entry" | awk '{print $1}')
    qa=$(echo "$entry" | awk '{print $2}')

    # repeat 0 to 1500 step qa
    for i in $(seq 0 $qa $total_qa); do
        randomSeed=$(shuf -i 1-100000 -n 1)
        num=$(($i / $qa))

        # With KVCACHE: Using CAG
        echo ""
        echo "[ HotpotQA $size $num, $i / $total_qa ]: Running KVCACHE using CAG"
        echo ""
        python ./kvcache.py --dataset "$dataset" --size "$size" --qa "$qa" \
            --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  "$randomSeed" \
            --kvcache file --similarity "bertscore" \
            --output "./results/new_results/${dataset}_${size}_kvcache_withCAG_qa_${qa}_rand_${randomSeed}.csv"

        # Without KVCACHE
        echo ""
        echo "[ HotpotQA $size $num, $i / $total_qa ]: Running KVCACHE not using CAG"
        echo ""
        python ./kvcache.py --dataset "$dataset" --size "$size" --qa "$qa" \
            --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  "$randomSeed" \
            --kvcache file --usePrompt --similarity "bertscore" \
            --output "./results/new_results/${dataset}_${size}_kvcache_noCAG_qa_${qa}_rand_${randomSeed}.csv"         

        for index in "${indices[@]}"; do
            for topk in "${top_k[@]}"; do
                echo ""
                echo "[ HotpotQA $size $num, $i / $total_qa ]: Running RAG with $index, topk ${topk}"
                echo ""
                python ./rag.py --dataset "$dataset" --size "$size" --qa "$qa" \
                    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  "$randomSeed" \
                    --index "$index" --topk "$topk" --similarity "bertscore" \
                    --output "./results/new_results/${dataset}_${size}_rag_${index}_${topk}_qa_${qa}_rand_${randomSeed}.csv" 
            done
        done

    done
done
