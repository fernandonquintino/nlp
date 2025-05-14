import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src import api_utils

api_utils.batch_size = 5
api_utils.backup_training_file = 'data/processed/labeled_output.csv'

labeled_path = "data/processed/labeled_output.csv"
raw_path = "data/raw/dataset_train.csv"

if os.path.exists(labeled_path):
    print("Found existing labeled data. Resuming...")
    api_utils.df = pd.read_csv(labeled_path)
else:
    print("No labeled data found. Starting from raw dataset...")
    api_utils.df = pd.read_csv(raw_path, sep='|')
    api_utils.df = api_utils.df.iloc[:, 1:]
    api_utils.df.rename(columns={'input;': 'text'}, inplace=True)
    api_utils.df['text'] = api_utils.df['text'].str.rstrip(';')

# Add rating columns if missing
model_list = list(api_utils.model_dict.keys())
for model in model_list:
    col = f"rating_{model}"
    if col not in api_utils.df.columns:
        api_utils.df[col] = pd.NA

# Label in parallel
with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
    executor.map(api_utils.run_batches_for_model, model_list)

# Save final labeled file
api_utils.df.to_csv(api_utils.backup_training_file, index=False)
print("Labeling complete.")