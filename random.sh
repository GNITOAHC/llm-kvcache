#!/bin/bash
logfilename="random.log"
# iteration of batch for each maxQuestion
iters=$(($maxQuestion/$batch))

# while log file exists, create a new one called random_i.log
i=1
while [ -f $logfilename ]; do
    echo "log file ${logfilename} exists, create a new one"
    logfilename="random$i.log"
    i=$(($i+1))
done

# 在這裡自訂 k 和 p 的值
k=3  # 設定 k 值
p=100  # 設定 p 值

datasets=("squad-train")
# models=("3.1-8B" "3.2-3B" "3.2-1B")
models=("3.1-8B")
indices=("openai" "bm25")
maxQuestions=("100" "1000")
top_k=("1" "3" "5" "10" "20")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for maxQuestion in "${maxQuestions[@]}"; do
      
      randomSeed=$(shuf -i 1-100000 -n 1)

      # Run KVCACHE without cache
      echo "Running KVCACHE for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model"
      python ./kvcache.py --kvcache file --dataset "$dataset" --similarity bertscore \
        --maxKnowledge "$k" --maxParagraph "$p" --maxQuestion "$maxQuestion" --usePrompt \
        --modelname "meta-llama/Llama-${model}-Instruct" --randomSeed  "$randomSeed" \
        --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_p${p}_q${maxQuestion}_${dataset}_bertscore_kvcache_nokv.txt"
      
      # Run KVCACHE
      echo "Running KVCACHE for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model"
      python ./kvcache.py --kvcache file --dataset "$dataset" --similarity bertscore \
        --maxKnowledge "$k" --maxParagraph "$p" --maxQuestion "$maxQuestion" \
        --modelname "meta-llama/Llama-${model}-Instruct" --randomSeed  "$randomSeed" \
        --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_p${p}_q${maxQuestion}_${dataset}_bertscore_kvcache.txt"
      
      # Run RAG
      for topk in "${top_k[@]}"; do
          for index in "${indices[@]}"; do
            echo "Running RAG with $index for $dataset, maxKnowledge $k, maxParagraph $p, maxQuestion $maxQuestion, model $model, topk ${topk}"
            python ./rag.py --index "$index" --dataset "$dataset" --similarity bertscore \
              --maxKnowledge "$k" --maxParagraph "$p" --maxQuestion "$maxQuestion" --topk "$topk" \
              --modelname "meta-llama/Llama-${model}-Instruct" --randomSeed  "$randomSeed" \
              --output "./results/${dataset}/${maxQuestion}/result_${model}_k${k}_p${p}_q${maxQuestion}_${dataset}_bertscore_rag_Index_${index}.txt_top${topk}" 
          done
      done
      
    done
  done
done


# datasets=("hotpotqa-train")
# # models=("3.1-8B" "3.2-3B" "3.2-1B")
# models=("3.1-8B")
# indices=("openai" "bm25")
# # maxQuestions=("10" "100")
# # maxQuestions=("32" "48" "64" "80")
# maxQuestions=("16" "32" "48" "64" "80")
# top_k=("1" "3" "5" "10" "20")
# # all k = 7405 article, tokens = 10,038,084 
# # when k = 16, tokens = 21,000
# # when k = 24, tokens = 32,667
# # when k = 32, tokens = 43,000
# # when k = 48, tokens = 64,000
# # when k = 64, tokens = 85,000
# # when k = 80, tokens = 106,000


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