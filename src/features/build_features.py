import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib

from src.features.feature_engineering import FeatureEngineer
from src.features.tfidf_features import TfidfFeatureExtractor
from src.features.keywords import CONTEXT_MARKERS


class FeatureBuilder:
    def __init__(self):
        self.engineer = FeatureEngineer(context_markers=CONTEXT_MARKERS)
        self.tfidf = TfidfFeatureExtractor()
        self.feature_columns = None

    def _prepare_tabular(self, df, fit=False):
        tabular = df.drop(columns=["content", "label"], errors="ignore")

        # FORCE NUMERIC ONLY (CRITICAL FIX)
        tabular = tabular.select_dtypes(include=[np.number])

        tabular = tabular.replace([np.inf, -np.inf], 0).fillna(0)

        if fit:
            self.feature_columns = tabular.columns.tolist()
        else:
            tabular = tabular.reindex(columns=self.feature_columns, fill_value=0)

        return csr_matrix(tabular.values.astype(np.float32))

    def fit(self, df):
        df = self.engineer.process(df.copy())

        texts = df["content"].astype(str).values
        X_text = self.tfidf.fit_transform(texts)

        X_tab = self._prepare_tabular(df, fit=True)

        X = hstack([X_text, X_tab]).tocsr()
        y = df["label"].astype(np.int32).values

        return X, y

    def transform(self, df):
        df = self.engineer.process(df.copy())

        texts = df["content"].astype(str).values
        X_text = self.tfidf.transform(texts)

        X_tab = self._prepare_tabular(df, fit=False)

        return hstack([X_text, X_tab]).tocsr()

    def fit_transform(self, df):
        return self.fit(df)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)