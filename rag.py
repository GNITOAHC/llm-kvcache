import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Document
import argparse
import os
from enum import Enum

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

# Define a simplified generate function
def generate(model, input_ids, max_length=1000):
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

from llama_index.core import Settings

def getOpenAIRetriever(documents: list[str]):
    """OpenAI RAG model"""
    import openai
    openai.api_key = get_env()["OPENAI_API_KEY"]        
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")
    
    from llama_index.embeddings.openai import OpenAIEmbedding
    # Set the embed_model in llama_index
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=get_env()["OPENAI_API_KEY"], title="openai-embedding")
    # model_name: "text-embedding-3-small", "text-embedding-3-large"
    
    # Create the OpenAI retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever()
    t2 = time()
    
    return OpenAI_retriever, t2 - t1
    

def getGeminiRetriever(documents: list[str]):
    """Gemini Embedding RAG model"""
    GOOGLE_API_KEY = get_env()["GOOGLE_API_KEY"]
    from llama_index.embeddings.gemini import GeminiEmbedding
    model_name = "models/embedding-001"
    # Set the embed_model in llama_index
    Settings.embed_model = GeminiEmbedding( model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")
    
    # Create the Gemini retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever()
    t2 = time()
    
    return Gemini_retriever, t2 - t1
    
def getBM25Retriever(documents: list[str]):
    from llama_index.core.node_parser import SentenceSplitter  
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    splitter = SentenceSplitter(chunk_size=512)
    
    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=2,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1

def get_dataset( filepath: str):
    df = pd.read_csv(filepath)
    dataset = zip(df['sample_question'], df['sample_ground_truth'])
    text_list = df["ki_text"].to_list()
    
    return text_list, dataset

def rag_test(args: argparse.Namespace):
    if args.dataset == "kis_sample":
        datapath = "./rag_sample_qas_from_kis.csv"
    if args.dataset == "kis":
        datapath = "./synthetic_knowledge_items.csv"
    
    text_list , dataset = get_dataset(datapath)
    
    # document indexing for the rag retriever
    documents = [Document(text=t) for t in text_list]
    
    if args.index == "gemini":
        retriever, prepare_time = getGeminiRetriever(documents)
    if args.index == "openai":
        retriever, prepare_time = getOpenAIRetriever(documents)
    if args.index == "bm25":
        retriever, prepare_time = getBM25Retriever(documents)
        
    print(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds\n")
    
    results = {
        "retrieve_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }
        
    for id, (question, ground_truth) in enumerate(dataset):
        # Retrieve the knowledge from the vector database
        retrieve_t1 = time()
        nodes = retriever.retrieve(question)
        retrieve_t2 = time()
        
        knowledge = nodes[0].text
        # short_knowledge = knowledge[:knowledge.find("**Step 4")]
        
        prompt = f"""
        {question}
        
        --- Retrieved Information ---
        {knowledge}
        --- End of Retrieved Information ---
        
        Using the retrieved information to answer the question.
        """

        # Generate Response for the question
        generate_t1 = time() 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output = generate(model, input_ids)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generate_t2 = time() 
        
        generated_text = generated_text.replace(prompt, "")
        
        # Evaluate bert-score similarity
        similarity = get_bert_similarity(generated_text, ground_truth)
        
        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t",
            f"retrieve time: {retrieve_t2 - retrieve_t1},\t",
            f"generate time: {generate_t2 - generate_t1}"
            )
        with open(args.output, "a") as f:
            f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t retrieve time: {retrieve_t2 - retrieve_t1},\t generate time: {generate_t2 - generate_t1}\n")
        
        results["prompts"].append(prompt)
        results["responses"].append(generated_text)
        results["retrieve_time"].append(retrieve_t2 - retrieve_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)
        
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_retrieve_time = sum(results["retrieve_time"]) / len(results["retrieve_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"retrieve time: {avg_retrieve_time},\t time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"retrieve time: {avg_retrieve_time},\t time: {avg_generate_time}\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    parser.add_argument('--index', choices=['gemini', 'openai', 'bm25'], required=True, help='Index to use (gemini, openai, bm25)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
    parser.add_argument('--dataset', choices=['kis', 'kis_sample'], required=True, help='Dataset to use (kis_sample, kis)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')

    args = parser.parse_args()
    
    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path
    
    if os.path.exists(args.output):
        args.output = unique_path(args.output)
        
    rag_test(args)