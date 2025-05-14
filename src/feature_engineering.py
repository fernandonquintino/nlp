import re
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np
from text_preprocessing import clean_text, correct_typos, spacy_lemmatize, remove_stopwords, light_clean_text_for_bert, enrich_text_full
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import BertTokenizer, BertForSequenceClassification


class FullFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, blocklist, whitelist,
                 trigram_group_mapping, bigram_group_mapping, trigram_list, bigram_list, use_light_clean_for_text=False):
        self.blocklist = blocklist
        self.whitelist = whitelist
        self.trigram_group_mapping = trigram_group_mapping
        self.bigram_group_mapping = bigram_group_mapping
        self.trigram_list = trigram_list
        self.bigram_list = bigram_list
        self.sia = SentimentIntensityAnalyzer()
        self.use_light_clean_for_text = use_light_clean_for_text
        
    def full_preprocess(self, text):
        if self.use_light_clean_for_text:
            text = light_clean_text_for_bert(text)
            return text  # For BERT, no further aggressive cleaning
        else:
            text = clean_text(text)
            text = correct_typos(text, self.blocklist, self.whitelist)[0]
            text = spacy_lemmatize(text)
            text = remove_stopwords(text)
            text = enrich_text_full(text,
                                    self.trigram_group_mapping,
                                    self.bigram_group_mapping,
                                    self.trigram_list,
                                    self.bigram_list)
            return text
        
    def extract_structured_features(self, text, final_text):
        has_number = int(bool(re.search(r"\d", text)))
        has_rating_number = int(bool(re.search(
            r"\b(?:give it a|rated?|score|rate|i give|i give this|its a|it's a|i rate|rating|^)(?:\s+)?(?:10|[0-9])\b",
            text, re.IGNORECASE)))
        text_length = len(final_text)
        n_tokens = len(final_text.split())
        sentiment = self.sia.polarity_scores(final_text)['compound']
        return [has_number, has_rating_number, text_length, n_tokens, sentiment]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        structured_features = []
        enriched_texts = []
        for text in X:
            final_text = self.full_preprocess(text)
            structured_features.append(self.extract_structured_features(text, final_text))
            enriched_texts.append(final_text)
        return np.array(structured_features), np.array(enriched_texts)

# Wrapper to select parts
class SelectStructured(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[0]  # structured part only

class SelectText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[1]  # enriched text part only