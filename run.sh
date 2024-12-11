#!/bin/bash

datasets=("hotpotqa-train" "hotpotqa-test")
# models=("3.1-8B" "3.2-3B" "3.2-1B")
models=("3.1-8B")
indices=("openai" "bm25")
# maxQuestions=("10" "100")
maxQuestions=("16" "24")
# top_k=("3" "1")
top_k=("1" "3")
# all k = 7405 article, tokens = 10,038,084 
# when k = 16, tokens = 21,000
# when k = 24, tokens = 31,500

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for maxQuestion in "${maxQuestions[@]}"; do
      k=$maxQuestion
      # Run KVCACHE
      echo "Running KVCACHE for $dataset, maxQuestion $maxQuestion, model $model"
      python ./kvcache.py --kvcache file --dataset "$dataset" --similarity bertscore \
        --maxKnowledge "$k" --maxQuestion "$maxQuestion" \
        --modelname "meta-llama/Llama-${model}-Instruct" \
        --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_q${maxQuestion}_${dataset}_bertscore_kvcache.txt"
      
      # Run RAG
      for topk in "${top_k[@]}"; do
          for index in "${indices[@]}"; do
            echo "Running RAG with $index for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model, topk ${topk}"
            python ./rag.py --index "$index" --dataset "$dataset" --similarity bertscore \
              --maxKnowledge "$k" --maxQuestion "$maxQuestion" --topk "$topk" \
              --modelname "meta-llama/Llama-${model}-Instruct" \
              --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_q${maxQuestion}_${dataset}_bertscore_rag_Index_${index}.txt_top${topk}" 
          done
      done
      
    done
  done
done
