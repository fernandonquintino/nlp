# api_utils.py

import os
import json
import time
import requests
import pandas as pd
from datetime import timedelta
from threading import Lock
from dotenv import load_dotenv

# Required global (create inside this file if not passed from notebook)

df = None  # will be assigned externally from notebook
batch_size = 20
backup_training_file = "labeled_output.csv"
save_lock = Lock()

# Load API key from .env
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("OPENROUTER_API_KEY")

# API endpoint and headers
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "SentimentClassifier"
}

# Define the 4 best models
model_dict = {
    "claude": "anthropic/claude-3.7-sonnet",
    "gpt4": "openai/gpt-4.1",
    # Those below are free but the api has a daily limit
    # "llama4": "meta-llama/llama-4-maverick:free",
    # "mistral": "mistralai/mistral-small-3.1-24b-instruct:free",
    
    # "deephermes": "nousresearch/deephermes-3-mistral-24b-preview:free",
    # "reka": "rekaai/reka-flash-3:free",
    # "deepseek": "deepseek/deepseek-chat-v3-0324:free"
}

# Main API call function
def transform_text_with_api(text, model_name="llama4"):
    model_id = model_dict.get(model_name)
    if not model_id:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(model_dict.keys())}")

    # Unified prompt for all models
    prompt = (
        "You are a sentiment classifier.\n"
        "Given a restaurant review, return only one of the following numbers:\n"
        "-1 for Negative\n"
        "0 for Neutral\n"
        "1 for Positive\n\n"
        f"Review: \"{text}\"\n\n"
        "Answer with just the number. No explanation. No punctuation."
    )

    # Single user message for all models
    messages = [{"role": "user", "content": prompt}]

    data = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 10
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        label = int(content.strip().split()[0])  # First word = numeric label
        return label
    except Exception as e:
        # print(f"[{model_name}] Error: {e}")
        # print("Raw response:", response.text if 'response' in locals() else "No response")
        return None

def call_with_retry(text, model_name, retries=3, retry_delay=5):
    for attempt in range(retries):
        result = transform_text_with_api(text, model_name)
        if result is not None:
            return result
        # print(f"[{model_name}] Retry {attempt + 1}/{retries} after rate limit.")
        time.sleep(retry_delay)
    # print(f"[{model_name}] Failed after {retries} retries.")
    return None

def run_batches_for_model(model_name):
    global df, batch_size, backup_training_file, save_lock
    
    # print(f"\nStarting model: {model_name}")

    # Set behavior for free models
    free_models = ["mistral", "llama4"]
    is_free_model = model_name in free_models
    delay_between_batches = 4 if is_free_model else 0
    use_retries = is_free_model

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.loc[start:end-1]
        mask = df.loc[start:end-1, f"rating_{model_name}"].isna()
        if not mask.any():
            continue

        # print(f"[{model_name}] Batch {start}-{end-1}")
        t0 = time.time()

        if use_retries:
            df.loc[mask.index, f"rating_{model_name}"] = batch.loc[mask.index, "text"].apply(
                lambda x: call_with_retry(x, model_name)
            )
        else:
            df.loc[mask.index, f"rating_{model_name}"] = batch.loc[mask.index, "text"].apply(
                lambda x: transform_text_with_api(x, model_name)
            )

        elapsed = timedelta(seconds=int(time.time() - t0))

        with save_lock:
            df.to_csv(backup_training_file, index=False)
            # print(f"[{model_name}] Saved batch {start}-{end-1}")
            # print(f"[{model_name}] Done in {elapsed}")

        if delay_between_batches > 0:
            time.sleep(delay_between_batches)
