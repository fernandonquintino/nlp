import os
import sys
sys.path.append(os.path.abspath('src'))

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer

from feature_engineering import FullFeatureExtractor, SelectStructured, SelectText
from src.nlp_models.sentence_bert_lr import SentenceBertStructuredLRClassifier, load_sentence_bert_lr
from src.nlp_models.fine_tuned_bert import FineTunedBertClassifier
from resources import blocklist, whitelist
from resources_lemmatization import trigram_group_mapping, bigram_group_mapping, trigram_list, bigram_list

def generate_all_model_predictions(X, pipeline_dummy, pipeline_bow_lr, pipeline_tfidf_lr, pipeline_tfidf_svd_lgbm, pipeline_lsa_lr, model_bert_ls_loaded, bert_clf_loaded):
    predictions = {
        'dummy': pipeline_dummy.predict(X),
        'bow_lr': pipeline_bow_lr.predict(X),
        'tfidf_lr': pipeline_tfidf_lr.predict(X),
        'tfidf_svd_lgbm': pipeline_tfidf_svd_lgbm.predict(X),
        'lsa_lr': pipeline_lsa_lr.predict(X),
        'bert_ls': model_bert_ls_loaded.predict(X),
        'bert_clf_loaded': np.array(bert_clf_loaded.predict(X))
    }

    return pd.DataFrame(predictions)

feature_extractor = FullFeatureExtractor(blocklist, whitelist,
                                         trigram_group_mapping, bigram_group_mapping,
                                         trigram_list, bigram_list)
feature_extractor_bert = FullFeatureExtractor(blocklist, whitelist,
                                         trigram_group_mapping, bigram_group_mapping,
                                         trigram_list, bigram_list, use_light_clean_for_text=True)

df_train = pd.read_csv('data/processed/final_label.csv')
df_test = pd.read_csv('data/raw/dataset_valid.csv', sep='|')
df_test = df_test.iloc[:, 1:]
df_test.rename(columns={'input': 'text'}, inplace=True)

X_train = df_train['text']
X_test = df_test['text']

pipeline_dummy = joblib.load('models/pipeline_dummy.joblib')
pipeline_bow_lr = joblib.load('models/pipeline_bow_lr.joblib')
pipeline_tfidf_lr = joblib.load('models/pipeline_tfidf_lr.joblib')
pipeline_tfidf_svd_lgbm = joblib.load('models/pipeline_tfidf_svd_lgbm.joblib')
pipeline_lsa_lr = joblib.load('models/pipeline_lsa_lr.joblib')

model_bert_ls_loaded = load_sentence_bert_lr('models/bert_sentence_lr', preprocessor=feature_extractor_bert)
bert_clf_loaded = FineTunedBertClassifier.load('models/bert_finetuned_model')

preds_train = generate_all_model_predictions(X_train, pipeline_dummy, pipeline_bow_lr, pipeline_tfidf_lr, pipeline_tfidf_svd_lgbm, pipeline_lsa_lr, model_bert_ls_loaded, bert_clf_loaded)
preds_test = generate_all_model_predictions(X_test, pipeline_dummy, pipeline_bow_lr, pipeline_tfidf_lr, pipeline_tfidf_svd_lgbm, pipeline_lsa_lr, model_bert_ls_loaded, bert_clf_loaded)

df_train = df_train.join(preds_train)
df_test = df_test.join(preds_test)
df_train['bert_clf_loaded'] = df_train['bert_clf_loaded'].astype(float)
df_test['bert_clf_loaded'] = df_test['bert_clf_loaded'].astype(float)

df_train.to_csv('data/results/results_train.csv', index=False)
df_test.to_csv('data/results/results_test.csv', index=False)

df_test[['text', 'bert_ls']].rename(columns={'bert_ls': 'prediction'}).to_csv('data/results/practical_results.csv', index=False)
df_test[['text', 'bert_clf_loaded']].rename(columns={'bert_clf_loaded': 'prediction'}).to_csv('data/results/best_results.csv', index=False)