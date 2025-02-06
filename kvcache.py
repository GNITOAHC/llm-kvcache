import torch
import argparse
import os
import cag.dataset as cagds
import cag.similarity as cagsim
from time import time
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import logging 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")


"""Hugging Face Llama model"""

global model_name, model, tokenizer
global rand_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


"""KV Cache test"""
# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


def generate(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int = 300
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        past_key_values: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, 
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            past_key_values = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if next_token.item() in model.config.eos_token_id:
                break
    return output_ids[:, origin_ids.shape[-1]:]


def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str,
) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    return outputs.past_key_values


def write_kv_cache(kv: DynamicCache, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def read_kv_cache(path: str) -> DynamicCache | None:
    """
    Read the KV Cache from a file. If the cache file is invalid or empty, return None.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        kv = torch.load(path, weights_only=True)
        return kv
    else:
        # Regenerate cache if it doesn't exist or is too small
        return None


def prepare_kvcache(documents, filepath: str = "./data_cache/cache_knowledges.pt", answer_instruction: str | None = None):
    # Prepare the knowledges kvcache

    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """
    # Get the knowledge cache
    t1 = time()
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    #print("kvlen: ", kv.key_cache[0].shape[-2])
    t2 = time()
    write_kv_cache(kv, filepath)
    logger.info(f"KV cache prepared in {t2 - t1:.2f} seconds.")
    return kv, t2 - t1

import csv
def kvcache_test(args: argparse.Namespace):
    answer_instruction = "Answer the question with a super short answer."
    text_list, dataset = cagds.get(args.dataset, args.size, args.qa, rand_seed)
    kvcache_path = "./data_cache/cache_knowledges.pt"


    knowledges = '\n\n\n\n\n\n'.join(text_list)
    knowledge_cache, prepare_time = prepare_kvcache(knowledges, filepath=kvcache_path, answer_instruction=answer_instruction)
    kv_len = knowledge_cache.key_cache[0].shape[-2]
    print(f"KVcache prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"KVcache prepared in {prepare_time} seconds\n")

    if args.usePrompt:
        results = ["idx", "generated_response", "ground_truth", "generated_time"]
    else:
        results = ["idx","generated_response","ground_truth","reset_cache_time","generate_time"]
    #idx, generated_response, ground_truth, reset_cache_time, generate_time

    dataset = list(dataset)  # Convert the dataset to a list

    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion is not None else len(dataset)
    # Retrieve the knowledge from the vector database
    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Generate Response for the question
        knowledges = '\n\n\n'.join(text_list)

        if args.usePrompt:
            prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are an assistant for giving short answers based on given context.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Context information is bellow.
            ------------------------------------------------
            {knowledges}
            ------------------------------------------------
            {answer_instruction}
            Question:
            {question}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
            generate_start = time()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, DynamicCache()) 
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
            generate_end = time()
            results.append([
                id,
                generated_text,
                ground_truth,
                generate_time
            ])
        else:
            prompt = f"""
            {question}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

            reset_start = time()
            clean_up(knowledge_cache, kv_len)
            reset_end = time()
            reset_cache_time = reset_end - reset_start
            generate_start = time()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, knowledge_cache)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
            generate_end = time()
            generate_time = generate_end - generate_start
            results.append([
                id,
                generated_text,
                ground_truth,
                reset_cache_time,
                generate_time
            ])


        # print("D: ", knowledges)
        print("Q: ", question)
        print("A: ", generated_text)
 
        # Evaluate bert-score similarity
        similarity = cagsim.bert(generated_text, ground_truth)

        with open(f"{args.size}_{args.dataset}_cag_{args.randomSeed}.csv", "w")  as f:
            writer = csv.writer(f)
            writer.writerows(results)


# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
)


def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True,     # Required for some models
        token=hf_token
    )

    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    parser.add_argument('--kvcache', choices=['file'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--usePrompt', default=False, action="store_true", help='Do not use cache')
    # 48 Articles, each article average 40~50 paragraph, each average 5~10 questions
    parser.add_argument('--modelname', required=False, default="meta-llama/Llama-3.2-1B-Instruct", type=str, help='Model name to use')
    parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
    parser.add_argument('--similarity', choices=['bertscore'], required=False, default="bertscore", help='Similarity metric to use')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=None ,type=int, help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', required=False, default=None ,type=int, help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None ,type=int, help='Maximum number of paragraph to use')
    parser.add_argument('--dataset', required=True, help='Dataset to use (kis, squad or hotpotqa)', 
                        choices=['kis', 'kis_sample', 
                                'squad-dev', 'squad-train', 
                                'hotpotqa-dev',  'hotpotqa-train', 'hotpotqa-test'])
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')

    # For new teseting
    parser.add_argument("--size", required=True, type=str, help="Dataset size to use", choices=["small", "medium", "large"])
    parser.add_argument("--qa", required=False, type=int, default=500, help="Total number of testing QA pairs")

    args = parser.parse_args()

    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion, "randomeSeed", args.randomSeed)

    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed is not None else None

    if args.quantized:
        tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN
        )

    def unique_path(path, i=0):
        if os.path.exists(path):
            # path = path.split("_")[:-1] if i != 0 else path
            return unique_path(path + "_" + str(i), i + 1)
        return path

    if os.path.exists(args.output):
        args.output = unique_path(args.output)

    kvcache_test(args)
