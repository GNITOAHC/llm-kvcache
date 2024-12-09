import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import argparse
import os
from time import time
import json

def get_env():
    env_dict = {}
    with open (file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

"""Hugging Face Llama model"""
HF_TOKEN = get_env()["HF_TOKEN"]
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

# Assume input_ids is your initial input sequence tensor, and max_length is the target length for decoding
# Define the maximum length for decoding
def generate(model, input_ids, past_key_values, max_length=50):
    output_ids = input_ids  # Start with initial input_ids
    eos_token_id = tokenizer.eos_token_id
    next_token_id = input_ids
    for _ in range(max_length - input_ids.size(1)):
        # Generate the model output using greedy decoding
        output = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
        logits = output.logits  # Extract the logits from the model output
        past_key_values = output.past_key_values  # Update past_key_values for the next step

        # Get the last token's logits and apply argmax to select the next token
        next_token_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)

        # Append the selected token to the output sequence
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

        # Stop if the end-of-sequence token is generated (assuming eos_token_id is defined)
        if next_token_id == eos_token_id:
            break
    return output_ids

"""KV Cache test"""
# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

def get_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return outputs.past_key_values

def write_kv_cache(kv: DynamicCache,path: str):
    torch.save(kv, path)

def read_kv_cache(path: str) -> DynamicCache:
    # kv = torch.load(path)
    kv = torch.load(path, weights_only=True)
    return kv

"""Sentence-BERT for evaluate semantic similarity"""
from sentence_transformers import SentenceTransformer, util
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight sentence-transformer

def get_bert_similarity(response, ground_truth):
    # Encode the query and text
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)

    # Compute the cosine similarity between the query and text
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)

    return cosine_score.item()

def prepare_kvcache(documents: list[str], filepath: str = "./data_cache/cache_knowledges.pt"):
    # Prepare the knowledges kvcache
    knowledges = ' '.join(documents)
    
    # Get the knowledge cache
    t1 = time()
    kv = get_kv_cache(model, tokenizer, knowledges)
    write_kv_cache(kv, filepath)
    t2 = time()
    
    return kv, t2 - t1
    
def get_kis_dataset(filepath: str):
    df = pd.read_csv(filepath)
    dataset = zip(df['sample_question'], df['sample_ground_truth'])
    text_list = df["ki_text"].to_list()
    
    return text_list, dataset

def parse_squad_data(raw):
    dataset = { "ki_text": [], "qas": [] }
    
    for k_id, data in enumerate(raw['data']):
        article = []
        for p_id, para in enumerate(data['paragraphs']):
            article.append(para['context'])
            for qa in para['qas']:
                ques = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                dataset['qas'].append({"title": data['title'], "paragraph": tuple((k_id, p_id)) ,"question": ques, "answers": answers})
        dataset['ki_text'].append({"title": data['title'], "paragraphs": article})
    
    return dataset

def get_squad_dataset(filepath: str):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = parse_squad_data(data)
    
    questions = [qa['question'] for qa in parsed_data['qas']]
    answers = [qa['answers'][0] for qa in parsed_data['qas']]
    dataset = zip(questions, answers)
    
    text_list = []
    for article in parsed_data['ki_text']:
        text_list.append(article['title'])
        text_list.extend(article['paragraphs'])
    
    return text_list, dataset
    
def kvcache_test(args: argparse.Namespace):
    if args.dataset == "kis_sample":
        datapath = "./datasets/rag_sample_qas_from_kis.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "kis":
        datapath = "./datasets/synthetic_knowledge_items.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "squad-dev":
        datapath = "./datasets/squad/dev-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath)
    if args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath)
    
    kvcache_path = "./data_cache/cache_knowledges.pt"
    documents_cache, prepare_time = prepare_kvcache(text_list, filepath=kvcache_path)
    
    print(f"KVcache prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"KVcache prepared in {prepare_time} seconds\n")
    
    results = {
        "cache_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }

    for id, (question, ground_truth) in enumerate(dataset):
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        
        # Read the knowledge cache from the cache file
        cache_t1 = time()
        if args.kvcache == "file":
            knowledge_cache = read_kv_cache(kvcache_path)
        # Not a good idea to use this method, as it will consume a lot of memory
        # if args.kvcache == "variable":
        #     knowledge_cache = documents_cache
        cache_t2 = time()
        
        # Generate Response for the question
        generate_t1 = time() 
        input_ids = tokenizer.encode( question , return_tensors="pt" ).to(model.device)
        output = generate(model, input_ids, knowledge_cache)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
        generate_t2 = time() 
        
        # Evaluate bert-score similarity
        similarity = get_bert_similarity(generated_text, ground_truth)
        
        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},",
            f"cache time: {cache_t2 - cache_t1},",
            f"generate time: {generate_t2 - generate_t1}"
            )
        with open(args.output, "a") as f:
            f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t cache time: {cache_t2 - cache_t1},\t generate time: {generate_t2 - generate_t1}\n")
        
        results["prompts"].append(question)
        results["responses"].append(generated_text)
        results["cache_time"].append(cache_t2 - cache_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)
        
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    # parser.add_argument('--kvcache', choices=['file', 'variable'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--kvcache', choices=['file'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
    parser.add_argument('--dataset', choices=['kis', 'kis_sample', 'squad-dev', 'squad-train'], required=True, help='Dataset to use (kis, kis_sample, squad-dev, squad-train)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')

    args = parser.parse_args()
    
    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path
    
    if os.path.exists(args.output):
        args.output = unique_path(args.output)
        
    kvcache_test(args)
