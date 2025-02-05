#!/bin/bash

datasets=("hotpotqa-train" "squad-train")
indices=("bm25")
sizes=("small" "medium" "large")
top_k=("3")
qa=("10")

randomSeed=$(shuf -i 1-100000 -n 1)

for dataset in "${datasets[@]}"; do
    for size in "${sizes[@]}"; do
        for index in "${indices[@]}"; do
            for topk in "${top_k[@]}"; do
                echo "Running RAG with $index for $dataset, model $model, size $size, topk ${topk}"
                python ./rag.py --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  "$randomSeed" \
                    --dataset "$dataset" --size "$size" --qa "$qa" \
                    --index "$index" --topk "$topk" --similarity "bertscore" \
                    --output "./results/new_results/${dataset}_${size}_${index}_${topk}_qa_${qa}_${randomSeed}.csv" 
            done
        done
    done
done
