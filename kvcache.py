import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

def get_env():
    env_dict = {}
    with open (".env", "r") as f:
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

# Define a simplified generate function
def generate(model, input_ids, max_length=50000):
    output_ids = input_ids  # Start with the initial input_ids
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_length - input_ids.size(1)):
        # Generate the model output using greedy decoding
        output = model(input_ids=output_ids)
        logits = output.logits  # Extract the logits from the model output

        # Get the last token's logits and apply argmax to select the next token
        next_token_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)

        # Append the selected token to the output sequence
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

        # Stop if the end-of-sequence token is generated
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

from time import time

def kv_cache_time():
    df = pd.read_csv("./rag_sample_qas_from_kis.csv")
    
    # Prepare the knowledges kvcache
    cache_dir = "./data_cache"
    knowledges = ' '.join(df['ki_text'])
    kv = get_kv_cache(model, tokenizer, knowledges)
    write_kv_cache(kv, f"{cache_dir}/cache_knowledges.pt")
    
    