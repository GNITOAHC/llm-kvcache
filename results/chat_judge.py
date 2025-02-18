# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go the next step.
3: If the Prediction exactly matches the Ground Truth, "score" is 1.
4: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
6: If the Prediction is self-contradictory, "score" must be 0.
7: If the prediction is not answering the question, "score" must be 0.
8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
10: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0."""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"score": 0, "explanation": "The prediction is not sure about the answer."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"score": 0, "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"score": 1, "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"score": 0, "explanation": "The prediction is not answering the question."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as w is the abbreviation of william."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as california quail is also known as california valley quail."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground truth: "india"
Prediction: "indianapolis"
Output: {"score": 0, "explanation": "The prediction doesn't match the ground truth answers."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"score": 1, "explanation": "The prediction correctly answers the result is a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"score": 1, "explanation": "The prediction matches ground truth as it correctly states the number of seconds asked in the question."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"score": 0, "explanation": "The prediction does not explicitly say the age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"score": 0, "explanation": "The prediction does not exactly match the ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"score": 1, "explanation": "The prediction answers the correct number of the year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"score": 1, "explanation": "The prediction gives the right number and it is reasonable."}
"""

def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


import datetime
import json
import os
import re
import time
from venv import logger
from openai import APIConnectionError, OpenAI, RateLimitError
import pandas as pd
import tqdm

def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response, 
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1

def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def evaluate_predictions(data_file):
    """
    Evaluates the predictions generated by a model against ground truth answers.
    
    Args:
    queries (List[str]): List of queries.
    ground_truths_list (List[List[str]]): List of lists of ground truth answers. 
        Note each query can have multiple ground truth answers.
    predictions (list): List of predictions generated by the model.
    evaluation_model_name (str): Name of the evaluation model.
    
    Returns:
    dict: A dictionary containing evaluation results.
    """
    # Load the data
    data = pd.read_csv(data_file)
    system_message = get_system_message()
    ## add a new column, gptscore
    data["gptscore"] = 0
    openai_client = OpenAI()

    with open(f"batches/{data_file}.jsonl", "w") as f:
        for idx, row in data.iterrows():
            question = row["question"]
            answer = row["generated_response"]
            ground_truth = row["ground_truth"]
            ground_truth_lowercase = ground_truth.lower()
            prediction_lowercase = answer.lower()
#{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
#{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

        #first write to batch files, then call the api       

            content = {
                "custom_id": f"{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4-0125-preview",
                    "messages": [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {question}\n Ground truth: {ground_truth}\n Prediction: {answer}\n",
                        },
                    ],
                    "max_tokens": 1000,
                }
            }
        
            f.write(json.dumps(content) + "\n")


    # Call the API

    batch_input_file = openai_client.files.create(
        file=open(f"batches/{data_file}.jsonl", "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    created_batch = openai_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    while True:
        batch = openai_client.batches.retrieve(created_batch.id)
        if batch.status == "completed":
            break
        time.sleep(1)
    file_response = openai_client.files.content(batch.output_file_id)
    #print(file_response.text)
    responses = file_response.text.split("\n")
    for idx, response in enumerate(responses):
        if response:
            explanation, score = parse_response(json.loads(response)["response"]['body']["choices"][0]['message']['content'])
            data.at[idx, "gptscore"] = score
    
    return data

directory = "results_w_question"
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm  # For progress tracking

def process_single_file(data_file):
    try:
        results = evaluate_predictions(f"results_w_question/{data_file}")
        results.to_csv(f"gpt_evaluated/{data_file}", index=False)
        return True
    except Exception as e:
        print(f"Error processing {data_file}: {str(e)}")
        return False

def process_files_parallel(directory, max_workers=12):
    # Create output directory if it doesn't exist
    os.makedirs("gpt_evaluated", exist_ok=True)
    
    # Get list of files
    data_files = os.listdir(directory)
    total_files = len(data_files)
    
    print(f"Processing {total_files} files using {max_workers} workers")
    
    # Process files in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and wrap with tqdm for progress tracking
        results = list(tqdm(
            executor.map(process_single_file, data_files),
            total=total_files,
            desc="Processing files"
        ))
    
    # Summary statistics
    successful = sum(results)
    failed = total_files - successful
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")

if __name__ == "__main__":
    process_files_parallel("results_w_question", max_workers=500)