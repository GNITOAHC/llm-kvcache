#!/bin/bash

# datasets=("hotpotqa-train")
# when k = 16, tokens = 21,000
# when k = 24, tokens = 32,667
# when k = 32, tokens = 43,000
# when k = 48, tokens = 64,000
# when k = 64, tokens = 85,000
# when k = 80, tokens = 106,000

# datasets=("squad-train")
# when k = 3, tokens = 21,000
# when k = 4, tokens = 32,000
# when k = 5, tokens = 50,000


# 在這裡自訂 k 和 p 的值
# k=3  # 設定 k 值
# p=100  # 設定 p 值

# datasets=("squad-dev")
# # models=("3.1-8B" "3.2-3B" "3.2-1B")
# models=("3.1-8B")
# indices=("openai" "bm25")
# maxQuestions=("10" "100")
# top_k=("3")

# for dataset in "${datasets[@]}"; do
#   for model in "${models[@]}"; do
#     for maxQuestion in "${maxQuestions[@]}"; do
      
#       # Run KVCACHE
#       echo "Running KVCACHE for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model"
#       python ./kvcache.py --kvcache file --dataset "$dataset" --similarity bertscore \
#         --maxKnowledge "$k" --maxParagraph "$p" --maxQuestion "$maxQuestion" \
#         --modelname "meta-llama/Llama-${model}-Instruct" \
#         --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_p${p}_q${maxQuestion}_${dataset}_bertscore_kvcache.txt"
      
#       # Run RAG
#       for topk in "${top_k[@]}"; do
#           for index in "${indices[@]}"; do
#             echo "Running RAG with $index for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model, topk ${topk}"
#             python ./rag.py --index "$index" --dataset "$dataset" --similarity bertscore \
#               --maxKnowledge "$k" --maxParagraph "$p" --maxQuestion "$maxQuestion" --topk "$topk" \
#               --modelname "meta-llama/Llama-${model}-Instruct" \
#               --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_p${p}_q${maxQuestion}_${dataset}_bertscore_rag_Index_${index}.txt_top${topk}" 
#           done
#       done
      
#     done
#   done
# done

