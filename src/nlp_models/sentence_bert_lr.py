import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class SentenceBertStructuredLRClassifier:
    def __init__(self, sentence_model_name_or_path, classifier=None, scaler=None, preprocessor=None):
        """
        - sentence_model_name_or_path: HuggingFace model name (e.g., 'paraphrase-MiniLM-L6-v2') or local path
        - classifier: Optional preloaded classifier (LogisticRegression)
        - scaler: Optional preloaded scaler (StandardScaler)
        - preprocessor: Your custom FullFeatureExtractor (must implement .transform(raw_texts) -> structured, cleaned_texts)
        """
        self.sentence_model = SentenceTransformer(sentence_model_name_or_path)
        self.classifier = classifier
        self.scaler = scaler
        self.preprocessor = preprocessor

    def fit(self, raw_texts, labels):
        """
        Trains the model.
        """
        # Preprocess: extract structured features and cleaned texts
        structured, cleaned_texts = self.preprocessor.transform(raw_texts)

        # Scale structured features
        self.scaler = StandardScaler()
        structured_scaled = self.scaler.fit_transform(structured)

        # Get sentence embeddings
        embeddings = self.sentence_model.encode(cleaned_texts)

        # Combine features
        combined = np.hstack([structured_scaled, embeddings])

        # Train classifier
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.classifier.fit(combined, labels)

    def predict(self, raw_texts):
        """
        Predicts using the trained model.
        """
        structured, cleaned_texts = self.preprocessor.transform(raw_texts)
        structured_scaled = self.scaler.transform(structured)
        embeddings = self.sentence_model.encode(cleaned_texts)
        combined = np.hstack([structured_scaled, embeddings])
        return self.classifier.predict(combined)

    def predict_proba(self, raw_texts):
        """
        Predicts probabilities.
        """
        structured, cleaned_texts = self.preprocessor.transform(raw_texts)
        structured_scaled = self.scaler.transform(structured)
        embeddings = self.sentence_model.encode(cleaned_texts)
        combined = np.hstack([structured_scaled, embeddings])
        return self.classifier.predict_proba(combined)

    def save(self, path):
        """
        Saves all components to the given path.
        """
        self.sentence_model.save(f'{path}/bert_sentence_model')
        joblib.dump(self.classifier, f'{path}/classifier.joblib')
        joblib.dump(self.scaler, f'{path}/scaler.joblib')
        # Preprocessor is usually stateless, not saved by default

    @classmethod
    def load(cls, path, preprocessor):
        """
        Loads the classifier from path. Requires preprocessor to be passed.
        """
        sentence_model_path = f'{path}/bert_sentence_model'
        classifier = joblib.load(f'{path}/classifier.joblib')
        scaler = joblib.load(f'{path}/scaler.joblib')
        return cls(sentence_model_name_or_path=sentence_model_path,
                   classifier=classifier, scaler=scaler, preprocessor=preprocessor)


def load_sentence_bert_lr(model_path, preprocessor):
    """
    Loads a SentenceBertStructuredLRClassifier from disk, fully ready to predict.

    Parameters:
    - model_path: Path to where the model was saved (e.g., 'models/bert_sentence_lr')
    - preprocessor: Your preprocessor (FullFeatureExtractor)

    Returns:
    - An instance of SentenceBertStructuredLRClassifier ready to predict
    """
    return SentenceBertStructuredLRClassifier.load(path=model_path, preprocessor=preprocessor)