import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self):
        self.label_map = {"Harmful": 1, "Neutral": 0}
        self.action_map = {"none": 0, "warning": 1, "removed": 2}

        self.city_mode = None
        self.post_type_mode = None
        self.date_median = None

    def validate_schema(self, df):
        required = ["content", "label", "user_reported", "action_taken", "city", "post_type", "date"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def fit(self, df):
        df = df.copy()

        df = self.validate_schema(df)

        self.city_mode = df["city"].mode(dropna=True)[0]
        self.post_type_mode = df["post_type"].mode(dropna=True)[0]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        self.date_median = df["date"].dropna().median()

        return self

    def clean_text(self, df):
        df = df.copy()
        df["content"] = df["content"].astype(str).str.strip()
        df["content"] = df["content"].replace("", np.nan)
        df["content"] = df["content"].fillna("missing_text")
        return df

    def encode_label(self, df):
        df = df.copy()
        df["label"] = df["label"].map(self.label_map)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df

    def encode_user_reported(self, df):
        df = df.copy()
        df["user_reported"] = df["user_reported"].fillna(0).astype(int)
        return df

    def encode_action_taken(self, df):
        df = df.copy()
        df["action_taken"] = (
            df["action_taken"]
            .fillna("none")
            .astype(str)
            .str.lower()
            .map(self.action_map)
            .fillna(0)
            .astype(int)
        )
        return df

    def normalize_text_fields(self, df):
        df = df.copy()

        df["city"] = df["city"].astype(str).str.lower().str.strip()
        df["post_type"] = df["post_type"].astype(str).str.lower().str.strip()

        df["city"] = df["city"].replace("nan", np.nan).fillna(self.city_mode)
        df["post_type"] = df["post_type"].replace("nan", np.nan).fillna(self.post_type_mode)

        return df

    def date_features(self, df):
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].fillna(self.date_median)

        df["post_year"] = df["date"].dt.year
        df["post_month"] = df["date"].dt.month
        df["post_dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["post_dayofweek"].isin([5, 6]).astype(int)

        reference_date = df["date"].max()
        df["days_since_post"] = (reference_date - df["date"]).dt.days

        return df.drop(columns=["date"])

    def transform(self, df):
        df = df.copy()

        df = self.clean_text(df)
        df = self.encode_label(df)
        df = self.encode_user_reported(df)
        df = self.encode_action_taken(df)
        df = self.normalize_text_fields(df)
        df = self.date_features(df)

        return df

    def process(self, df):
        return self.transform(df)