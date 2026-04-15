from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TfidfFeatureExtractor:
    def __init__(self, max_features=4000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def save(self, path):
        joblib.dump(self.vectorizer, path)

    def load(self, path):
        self.vectorizer = joblib.load(path)
        return self