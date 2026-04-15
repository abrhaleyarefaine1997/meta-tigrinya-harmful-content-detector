import xgboost as xgb
import pandas as pd
import joblib


class Predictor:
    def __init__(self, model_path, feature_builder_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        self.feature_builder = joblib.load(feature_builder_path)

    def _build_input(self, text):
        return pd.DataFrame({
            "content": [text],
            "user_reported": [0],
            "action_taken": ["none"],
            "city": ["unknown"],
            "post_type": ["unknown"],
            "date": [pd.Timestamp.today()]
        })

    def predict_proba(self, text):
        df = self._build_input(text)

        X = self.feature_builder.transform(df)

        dmatrix = xgb.DMatrix(X)

        return float(self.model.predict(dmatrix)[0])

    def predict(self, text):
        return int(self.predict_proba(text) >= 0.5)