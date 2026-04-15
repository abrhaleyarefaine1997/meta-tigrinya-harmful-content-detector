import pandas as pd


class DatasetLoader:
    def __init__(self, path):
        self.path = path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.path)
        return self.df

    def validate(self):
        if self.df is None:
            raise ValueError("Dataset not loaded")

        required_columns = [
            "content",
            "label",
            "post_type",
            "user_reported",
            "action_taken",
            "city",
            "date"
        ]

        missing = [col for col in required_columns if col not in self.df.columns]

        if len(missing) > 0:
            raise ValueError(f"Missing columns: {missing}")

    def get_raw_data(self):
        if self.df is None:
            raise ValueError("Dataset not loaded")

        return self.df.copy()