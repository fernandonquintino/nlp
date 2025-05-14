import os
import pandas as pd
import requests
from dotenv import load_dotenv

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
    "X-Title": "TypoWhitelistChecker"
}

# Supported models
model_dict = {
    "gpt4": "openai/gpt-4.1"
}

# Function to call the API
def check_correction(text, correction, model_name="gpt4"):
    model_id = model_dict.get(model_name)
    if not model_id:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(model_dict.keys())}")

    prompt = (
        f"You are a spelling correction assistant.\n\n"
        f"Original sentence:\n{text}\n\n"
        f"Suggested correction:\n{correction}\n\n"
        f"Does the correction make sense? Answer only with '1' if yes or '0' if no."
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"].strip()

    # Extract only '1' or '0'
    return 1 if '1' in result else 0

# Main function to process dataframe
def generate_whitelist(df, text_column='text', model_name="gpt4"):
    all_results = []

    for _, row in df.iterrows():
        sentence = row[text_column]
        corrections = [c.strip() for c in row['corrections'].split(';') if c.strip()]
        row_results = []

        for correction in corrections:
            result = check_correction(sentence, correction, model_name=model_name)
            row_results.append(result)

        all_results.append(row_results)

    return all_results





# Function to call the API
def check_bigram_relevance(bigram, freq=None, pmi_score=None, model_name="gpt4"):
    model_id = model_dict.get(model_name)
    if not model_id:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(model_dict.keys())}")

    prompt = (
        f"You are an NLP assistant.\n\n"
        f"Here is a bigram:\n'{bigram}'\n"
        f"It appeared {freq} times (if known), and has a PMI score of {pmi_score}.\n\n"
        f"Should this bigram be kept as a single token (unigram) for a machine learning model? "
        f"Consider if the bigram adds specific meaning that would be lost if using only its parts, "
        f"or if the unigram already carries enough meaning.\n\n"
        f"Answer only with '1' (keep) or '0' (discard)."
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"].strip()

    return 1 if '1' in result else 0

def generate_bigram_decision(df, model_name="gpt4"):
    decisions = []

    for _, row in df.iterrows():
        bigram = row['bigram']
        freq = row.get('frequency_x') if not pd.isna(row.get('frequency_x')) else row.get('frequency_y')
        pmi_score = row['pmi_score']

        result = check_bigram_relevance(bigram, freq, pmi_score, model_name=model_name)
        decisions.append(result)

    return decisions
