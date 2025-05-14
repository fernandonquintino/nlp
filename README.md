# NLP Review Classification Pipeline

This project builds a **reproducible, modular pipeline for restaurant review classification** using the following models:

* dummy (always predicts 1)
* bow_lr (Bag-of-words + Logistic Regression)
* tfidf_lr (Term Frequency-Inverse Document Frequency + Structured + Logistic Regression)
* tfidf_svd_lgbm (Term Frequency-Inverse Document Frequency + Singular Value Decomposition + LightGBM)
* lsa_lr (Latent Semantic Analysis + Logistic Regression)
* bert_ls (BERT embeddings + Logistic Regression)
* bert_clf (Fine-tuned BERT)

It includes all steps from **labeling, preprocessing, enrichment, feature engineering, and model training**, managed via a **single `main.py` script**.

## Project Structure
```
project/
├── data/ # Raw and processed datasets (also uses a dictionary)
│ └── dictionaries/
│ └── processed/
│ └── raw/ # Raw data
│ └── results/ # Predictions
├── models/ # Trained models
├── notebooks/ # Notebooks with EDA
├── 1_label_target.ipynb # API-based or heuristic labeling
├── 2_fine_tuning_label.ipynb # Refines labels
├── 3_processing_step.ipynb # Preprocessing
├── 4_eda.ipynb # EDA
├── 5_metrics.ipynb # Calculate metrics using cross-validation
├── 6_pipeline_models.ipynb # Train complete pipelines (preprocessing + modelling)
├── 7_predictions.ipynb # Predictions
├── src/ # Source code 
│ ├── api_utils.py
│ ├── ensure_resources.py
│ ├── feature_engineering.py
│ ├── llm_preprocessing.py
│ ├── resources_lemmatization.py
│ ├── resources.py
│ ├── text_preprocessing.py
│ └── nlp_models/
├── 1_label_target.py # API-based or heuristic labeling
├── 2_fine_tuning_label.py # Refines labels
├── 3_preprocessing.py # Text cleaning, typos correction
├── 4_lemmatization.py # Text enrichment, feature generation
├── 5_train_models.py # Train all models
├── main.py # Runs the full pipeline (venv + requirements + steps)
└── requirements.txt
```
## Row to run

### Clone the repository
```bash
git clone git@github.com:fernandonquintino/nlp-review-classification.git
cd nlp-review-classification
```

### Retrain everything:
* Open a terminal (ensure you are in the main folder) and type python main.py

### Make predictions
* Open notebook 7_predictions.ipynb (it loads the models, make predictions and also save the data on the results file)
* Execute 6_predictions.py (also executed in the main.py)

## Important observations

* This project uses OpenRouter API and paid models. You have to create a .env file: with the following OPENROUTER_API_KEY=123abc (place your key here)
* For practical use the suggested model is the BERT Sentence Embeddings + Logistic Regression. If more time or processing is available, then the suggestions if to use Fine-tuned BERT
* Results for all models are in the data/results/ folder. All results are in the file results_test.csv. Best_results.csv (fine-tuned bert) and practical_results.csv (bert embedding + lr) are for direct conference (columns: text/prediction)
* Raw data has | as separators. It should be loaded in a ipynb file as:

 ```
df_test = pd.read_csv('data/raw/dataset_valid.csv', sep='|') # (same for dataset_train.csv)
df_test = df_test.iloc[:, 1:]
df_test.rename(columns={'input': 'text'}, inplace=True)
```