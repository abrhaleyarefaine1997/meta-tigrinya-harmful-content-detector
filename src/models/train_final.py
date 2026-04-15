import os
import xgboost as xgb


def train_final_model(X, y):
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500
    )

    return model


def save_artifacts(model, feature_builder, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    model.save_model(os.path.join(model_dir, "xgb_model.json"))
    feature_builder.save(os.path.join(model_dir, "feature_builder.pkl"))