#!/bin/bash
logfilename="./log/random-hotpot.log"
# while log file exists, create a new one called random_i.log
i=1
while [ -f $logfilename ]; do
    echo "log file ${logfilename} exists, create a new one"
    logfilename="./log/random-hotpot$i.log"
    i=$(($i+1))
done

# # all k = 7405 article, tokens = 10,038,084 
# # when k = 1, tokens = 1,400
# # when k = 16, tokens = 22,400
# # when k = 24, tokens = 33,667
# # when k = 32, tokens = 44,800
# # when k = 48, tokens = 64,000
# # when k = 64, tokens = 85,000
# # when k = 80, tokens = 106,000

datasets=("hotpotqa-train")
# models=("3.1-8B" "3.2-3B" "3.2-1B")
models=("3.1-8B")
indices=("openai" "bm25")
maxQuestions=("16" "32" "48" "64" "80")
top_k=("1" "3" "5" "10" "20")


for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for maxQuestion in "${maxQuestions[@]}"; do
      k=$maxQuestion

      randomSeed=$(shuf -i 1-100000 -n 1)

      # Run KVCACHE without cache
      echo "Running KVCACHE for $dataset, maxQuestion $maxQuestion, model $model"
      python ./kvcache.py --kvcache file --dataset "$dataset" --similarity bertscore \
        --maxKnowledge "$k" --maxQuestion "$maxQuestion" --usePrompt \
        --modelname "meta-llama/Llama-${model}-Instruct" --randomSeed  "$randomSeed" \
        --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_q${maxQuestion}_${dataset}_bertscore_kvcache_nokv.txt"

      # Run KVCACHE
      echo "Running KVCACHE for $dataset, maxQuestion $maxQuestion, model $model"
      python ./kvcache.py --kvcache file --dataset "$dataset" --similarity bertscore \
        --maxKnowledge "$k" --maxQuestion "$maxQuestion" \
        --modelname "meta-llama/Llama-${model}-Instruct" --randomSeed  "$randomSeed" \
        --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_q${maxQuestion}_${dataset}_bertscore_kvcache.txt"
      
      # Run RAG
      for topk in "${top_k[@]}"; do
          for index in "${indices[@]}"; do
            echo "Running RAG with $index for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model, topk ${topk}"
            python ./rag.py --index "$index" --dataset "$dataset" --similarity bertscore \
              --maxKnowledge "$k" --maxQuestion "$maxQuestion" --topk "$topk" \
              --modelname "meta-llama/Llama-${model}-Instruct" --randomSeed  "$randomSeed" \
              --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_q${maxQuestion}_${dataset}_bertscore_rag_Index_${index}.txt_top${topk}" 
          done
      done
      
    done
  done
done
