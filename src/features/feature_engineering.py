import pandas as pd
import numpy as np
import re


class FeatureEngineer:
    def __init__(self, context_markers=None):
        self.context_markers = list(set([
            str(k).strip().lower()
            for k in (context_markers or [])
            if str(k).strip()
        ]))

        self.pattern = (
            re.compile("|".join(map(re.escape, self.context_markers)))
            if self.context_markers else None
        )

    def text_features(self, df):
        df["content"] = df["content"].astype(str)

        df["word_count"] = df["content"].str.split().apply(len)
        df["char_count"] = df["content"].str.len()

        df["avg_word_length"] = (
            df["char_count"] / (df["word_count"] + 1)
        ).replace([np.inf, -np.inf], 0).fillna(0)

        df["has_link"] = df["content"].str.contains(r"http[s]?://", na=False).astype(int)
        df["has_number"] = df["content"].str.contains(r"\d", na=False).astype(int)

        df["uppercase_ratio"] = df["content"].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )

        df["punctuation_count"] = df["content"].str.count(r"[^\w\s]")

        return df

    def harmful_features(self, df):
        if self.pattern is None:
            df["has_harmful_keyword"] = 0
            df["harmful_keyword_count"] = 0
            return df

        df["has_harmful_keyword"] = df["content"].apply(
            lambda x: int(bool(self.pattern.search(str(x).lower())))
        )

        df["harmful_keyword_count"] = df["content"].apply(
            lambda x: len(self.pattern.findall(str(x).lower()))
        )

        return df

    def behavioral_features(self, df):
        df["user_is_reported"] = df["user_reported"].fillna(0).astype(int)
        df["action_flagged"] = (df["action_taken"].astype(str).str.lower() != "none").astype(int)
        return df

    def interaction_features(self, df):
        df["keyword_and_reported"] = (
            (df["has_harmful_keyword"] == 1) &
            (df["user_is_reported"] == 1)
        ).astype(int)

        df["harmful_no_action"] = (
            (df["has_harmful_keyword"] == 1) &
            (df["action_flagged"] == 0)
        ).astype(int)

        df["no_keyword_but_action"] = (
            (df["has_harmful_keyword"] == 0) &
            (df["action_flagged"] == 1)
        ).astype(int)

        df["report_action_mismatch"] = (
            (df["user_is_reported"] == 1) &
            (df["action_flagged"] == 0)
        ).astype(int)

        return df

    def frequency_features(self, df):
        if "city" in df.columns:
            city_counts = df["city"].value_counts()
            df["city_freq"] = df["city"].map(city_counts).fillna(0)
        return df

    def categorical_features(self, df):
        if "post_type" in df.columns:
            df = pd.get_dummies(df, columns=["post_type"], prefix="type")
        return df

    def length_features(self, df):
        df["length_very_short"] = (df["word_count"] < 5).astype(int)
        df["length_short"] = ((df["word_count"] >= 5) & (df["word_count"] < 15)).astype(int)
        df["length_medium"] = ((df["word_count"] >= 15) & (df["word_count"] < 40)).astype(int)
        df["length_long"] = (df["word_count"] >= 40).astype(int)
        return df

    def process(self, df):
        df = df.copy()

        df = self.text_features(df)
        df = self.harmful_features(df)
        df = self.behavioral_features(df)
        df = self.interaction_features(df)
        df = self.frequency_features(df)
        df = self.categorical_features(df)
        df = self.length_features(df)

        return df