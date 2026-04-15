import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.evaluation import ModelEvaluator


def cross_validate_xgb(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []

    best_model = None
    best_auc = -1

    evaluator = ModelEvaluator()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

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
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        probs = model.predict(dval)
        preds = (probs >= 0.5).astype(int)

        f1 = f1_score(y_val, preds)
        precision = precision_score(y_val, preds)
        recall = recall_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        auc_scores.append(auc)

        print(
            f"Fold {fold} | "
            f"F1={f1:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"AUC={auc:.4f}"
        )

        evaluator.plot_confusion_matrix(
            y_val, preds, name=f"cm_fold_{fold}"
        )

        evaluator.plot_roc_curve(
            y_val, probs, name=f"roc_fold_{fold}"
        )

        evaluator.plot_pr_curve(
            y_val, probs, name=f"pr_fold_{fold}"
        )

        if auc > best_auc:
            best_auc = auc
            best_model = model

    print("\n=== CROSS VALIDATION SUMMARY ===")
    print(f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"AUC: {np.mean(auc_scores):.4f}")

    return best_model