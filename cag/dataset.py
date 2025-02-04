import json
import random
import pandas as pd
from typing import Iterator

def _parse_squad_data(raw):
    dataset = {"ki_text": [], "qas": []}

    for k_id, data in enumerate(raw["data"]):
        article = []
        for p_id, para in enumerate(data["paragraphs"]):
            article.append(para["context"])
            for qa in para["qas"]:
                ques = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                dataset["qas"].append(
                    {
                        "title": data["title"],
                        "paragraph_index": tuple((k_id, p_id)),
                        "question": ques,
                        "answers": answers,
                    }
                )
        dataset["ki_text"].append(
            {"id": k_id, "title": data["title"], "paragraphs": article}
        )

    return dataset


def squad(
    filepath: str,
    max_knowledge: int | None = None,
    max_paragraph: int | None = None,
    max_questions: int | None = None,
    random_seed: int | None = None,
) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @param max_knowledge: maximum number of docs in dataset
    @param max_paragraph:
    @param max_questions:
    @return: knowledge list, question & answer pair list
    """
    # Open and read the JSON file
    with open(filepath, "r") as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = _parse_squad_data(data)

    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = (
        max_knowledge
        if max_knowledge is not None and max_knowledge < len(parsed_data["ki_text"])
        else len(parsed_data["ki_text"])
    )
    max_paragraph = max_paragraph if max_knowledge == 1 else None

    # Shuffle the Articles and Questions
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])

    k_ids = [i["id"] for i in parsed_data["ki_text"][:max_knowledge]]

    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data["ki_text"][:max_knowledge]:
        max_para = (
            max_paragraph
            if max_paragraph is not None and max_paragraph < len(article["paragraphs"])
            else len(article["paragraphs"])
        )
        text_list.append(article["title"])
        text_list.append("\n".join(article["paragraphs"][0:max_para]))

    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [
        qa["question"]
        for qa in parsed_data["qas"]
        if qa["paragraph_index"][0] in k_ids
        and (max_paragraph is None or qa["paragraph_index"][1] < max_paragraph)
    ]
    answers = [
        qa["answers"][0]
        for qa in parsed_data["qas"]
        if qa["paragraph_index"][0] in k_ids
        and (max_paragraph is None or qa["paragraph_index"][1] < max_paragraph)
    ]

    dataset = zip(questions, answers)

    return text_list, dataset[:max_questions]


def hotpotqa(
    filepath: str, 
    max_knowledge: int | None = None, 
    max_questions: int | None = None,
    random_seed: int | None = None
) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @param max_knowledge:
    @return: knowledge list, question & answer pair list
    """
    # Open and read the JSON
    with open(filepath, "r") as file:
        data = json.load(file)

    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(data)

    questions = [qa["question"] for qa in data]
    answers = [qa["answer"] for qa in data]
    dataset = zip(questions, answers)

    if max_knowledge is None:
        max_knowledge = len(data)
    else:
        max_knowledge = min(max_knowledge, len(data))

    text_list = []
    for _, qa in enumerate(data[:max_knowledge]):
        context = qa["context"]
        context = [c[0] + ": \n" + "".join(c[1]) for c in context]
        article = "\n\n".join(context)

        text_list.append(article)

    return text_list, dataset[:max_questions]


def kis(filepath: str) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @return: knowledge list, question & answer pair list
    """
    df = pd.read_csv(filepath)
    dataset = zip(df["sample_question"], df["sample_ground_truth"])
    text_list = df["ki_text"].to_list()

    return text_list, dataset


def get(
    dataset: str,
    size: str = None,
    qa_pairs: int = None,
    random_seed: int = None,
) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param dataset: dataset name
    @param size: dataset size
    @param qa_pairs: number of question & answer pairs
    """
    
    if size is not in ["small", "medium", "large"]:
        raise ValueError("size must be one of 'small', 'medium', 'large'")
    
    if dataset.startswith("squad") and size is None:
        max_paragraph = 0
        max_knowledge = {
            "small": 3,  # 3 docs ≈ 21k tokens
            "medium": 4, # 4 docs ≈ 32k tokens
            "large": 7   # 7 docs ≈ 50k tokens
        }.get(size, None)

    if dataset.startswith("hotpotqa") and size is None:
        max_knowledge = {
            "small": 16,  # 16 docs ≈ 21k tokens  
            "medium": 32, # 32 docs ≈ 43k tokens  
            "large": 64   # 64 docs ≈ 85k tokens  
        }.get(size, None)
        
    match dataset:
        case "kis_sample":
            path = "./datasets/rag_sample_qas_from_kis.csv"
            return kis(path)
        case "kis":
            path = "./datasets/synthetic_knowledge_items.csv"
            return kis(path)
        case "squad-dev":
            path = "./datasets/squad/dev-v1.1.json"
            return squad(path, max_knowledge, max_paragraph, qa_pairs, random_seed)
        case "squad-train":
            path = "./datasets/squad/train-v1.1.json"
            return squad(path, max_knowledge, max_paragraph, qa_pairs, random_seed)
        case "hotpotqa-dev":
            path = "./datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
            return hotpotqa(path, max_knowledge, qa_pairs, random_seed)
        case "hotpotqa-test":
            path = "./datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
            return hotpotqa(path, max_knowledge, qa_pairs, random_seed)
        case "hotpotqa-train":
            path = "./datasets/hotpotqa/hotpot_train_v1.1.json"
            return hotpotqa(path, max_knowledge, qa_pairs, random_seed)
        case _:
            return [], zip([], [])
