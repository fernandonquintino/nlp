import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.abspath('src'))
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from lightgbm import LGBMClassifier

from src.feature_engineering import FullFeatureExtractor, SelectStructured, SelectText
from src.nlp_models.sentence_bert_lr import SentenceBertStructuredLRClassifier
from src.nlp_models.fine_tuned_bert import FineTunedBertClassifier

from src.resources import blocklist, whitelist
from src.resources_lemmatization import trigram_group_mapping, bigram_group_mapping, trigram_list, bigram_list


feature_extractor = FullFeatureExtractor(blocklist, whitelist,
                                         trigram_group_mapping, bigram_group_mapping,
                                         trigram_list, bigram_list)
feature_extractor_bert = FullFeatureExtractor(blocklist, whitelist,
                                         trigram_group_mapping, bigram_group_mapping,
                                         trigram_list, bigram_list, use_light_clean_for_text=True)

df = pd.read_csv('data/processed/final_label.csv')
X = df['text']
y = df['target']

# Train models

## Dummy model

structured_pipeline = Pipeline([('select_structured', SelectStructured()),
                                ('scaler', StandardScaler())])
pipeline_dummy = Pipeline([('feature_extractor', feature_extractor),
                           ('structured', structured_pipeline),
                           ('clf', DummyClassifier(strategy='most_frequent'))])
pipeline_dummy.fit(X, y)


## BOW + Logistic Regression

text_pipeline = Pipeline([('select_text', SelectText()),
                          ('bow', CountVectorizer())])
structured_pipeline = Pipeline([('select_structured', SelectStructured()),
                                ('scaler', StandardScaler())])
combined_features = FeatureUnion([('structured', structured_pipeline),
                                  ('text', text_pipeline)])
pipeline_bow_lr = Pipeline([('feature_extractor', feature_extractor),
                            ('combined_features', combined_features),
                            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
pipeline_bow_lr.fit(X, y)


## TF-IDF + Structured + Logistic Regression

text_pipeline_tfidf = Pipeline([('select_text', SelectText()),
                                ('tfidf', TfidfVectorizer())])
combined_features_tfidf = FeatureUnion([('structured', structured_pipeline),
                                        ('text', text_pipeline_tfidf)])
pipeline_tfidf_lr = Pipeline([('feature_extractor', feature_extractor),
                              ('combined_features', combined_features_tfidf),
                              ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
pipeline_tfidf_lr.fit(X, y)


## TF-IDF + SVD + LightGBM

text_pipeline_tfidf_svd = Pipeline([('select_text', SelectText()),
                                    ('tfidf', TfidfVectorizer()),
                                    ('svd', TruncatedSVD(n_components=100, random_state=42))])

combined_features_tfidf_svd = FeatureUnion([('structured', structured_pipeline),
                                            ('text', text_pipeline_tfidf_svd)])

pipeline_tfidf_svd_lgbm = Pipeline([('feature_extractor', feature_extractor),
                                    ('combined_features', combined_features_tfidf_svd),
                                    ('clf', LGBMClassifier(num_leaves=15, max_depth=3, learning_rate=0.05,  n_estimators=200, class_weight='balanced', random_state=42, verbose=-1))])
pipeline_tfidf_svd_lgbm.fit(X, y)


## LSA (TF-IDF + SVD) + Logistic Regression

text_pipeline_lsa = Pipeline([('select_text', SelectText()),
                              ('tfidf', TfidfVectorizer()),
                              ('svd', TruncatedSVD(n_components=100, random_state=42))])
combined_features_lsa = FeatureUnion([('structured', structured_pipeline),
                                      ('text', text_pipeline_lsa)])
pipeline_lsa_lr = Pipeline([('feature_extractor', feature_extractor),
                            ('combined_features', combined_features_lsa),
                            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])

pipeline_lsa_lr.fit(X, y)


## BERT Sentence Embeddings + Logistic Regression

model_bert_ls = SentenceBertStructuredLRClassifier(sentence_model_name_or_path='paraphrase-MiniLM-L6-v2', preprocessor=feature_extractor_bert)
model_bert_ls.fit(X, y)


## Fine-tuned bert

bert_clf = FineTunedBertClassifier('bert-base-uncased')
bert_clf.fit(X, y, epochs=3)


# Save models

joblib.dump(pipeline_bow_lr, 'models/pipeline_bow_lr.joblib')
joblib.dump(pipeline_tfidf_lr, 'models/pipeline_tfidf_lr.joblib')
joblib.dump(pipeline_tfidf_svd_lgbm, 'models/pipeline_tfidf_svd_lgbm.joblib')
joblib.dump(pipeline_lsa_lr, 'models/pipeline_lsa_lr.joblib')
joblib.dump(pipeline_dummy, 'models/pipeline_dummy.joblib')
model_bert_ls.save('models/bert_sentence_lr')
bert_clf.save('models/bert_finetuned_model')
