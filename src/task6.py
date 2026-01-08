# src/task6.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

# Optional xgboost
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# -----------------------
# Config (same as your main)
# -----------------------
TASK6_NUM_FEATURES = ["Age", "noofpatients", "noofinvestigation", "nooftreatment"]
TASK6_CAT_FEATURES = ["Period", "DayofWeek", "HRG"]


# -----------------------
# Data helpers (same logic as your main)
# -----------------------
def ensure_breach_flag(
    df: pd.DataFrame,
    los_col: str = "LoS",
    breach_col: str = "Breachornot",
    out_col: str = "Breach_flag",
    breach_minutes: int = 240,
) -> pd.DataFrame:
    """
    Robust breach flag:
    - if Breachornot exists: map text/num -> 0/1
    - else fallback to LoS > 240 if LoS exists
    """
    d = df.copy()

    if breach_col in d.columns:
        s = d[breach_col]
        if pd.api.types.is_numeric_dtype(s):
            d[out_col] = (pd.to_numeric(s, errors="coerce") > 0).astype("Int64")
        else:
            ss = s.astype(str).str.strip().str.lower()
            d[out_col] = ss.map({
                "breach": 1,
                "non-breach": 0, "non breach": 0, "no breach": 0,
                "1": 1, "0": 0,
                "true": 1, "false": 0,
                "yes": 1, "no": 0,
                "y": 1, "n": 0,
            }).astype("Float64")
            d[out_col] = pd.to_numeric(d[out_col], errors="coerce")
    else:
        if los_col not in d.columns:
            raise ValueError(f"Missing both {breach_col} and {los_col}, cannot create breach flag.")
        d[out_col] = (pd.to_numeric(d[los_col], errors="coerce") > breach_minutes).astype("Int64")

    d = d.dropna(subset=[out_col]).copy()
    d[out_col] = d[out_col].astype(int)
    return d


def build_Xy_no_los(
    df: pd.DataFrame,
    num_features: List[str],
    cat_features: List[str],
    target_col: str = "Breach_flag",
) -> Tuple[pd.DataFrame, pd.Series]:
    need = num_features + cat_features + [target_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[Task6] Missing columns: {missing}. Available: {list(df.columns)}")

    X = df[num_features + cat_features].copy()
    y = df[target_col].copy()

    # numeric
    for c in num_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median()
        X[c] = X[c].fillna(med)

    # categorical
    for c in cat_features:
        X[c] = X[c].astype(str)
        X[c] = X[c].replace({"nan": "Missing", "None": "Missing"}).fillna("Missing")

    return X, y


def make_preprocess(cat_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough",
    )


# -----------------------
# Trained bundle (so threshold can change without retraining)
# -----------------------
@dataclass
class Task6TrainedBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    models: Dict[str, Pipeline]            # fitted pipelines
    y_prob: Dict[str, np.ndarray]          # test probabilities per model
    seed: int
    test_size: float


def train_task6_models(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.25,
    use_xgb: bool = True,
) -> Task6TrainedBundle:
    d = ensure_breach_flag(df, out_col="Breach_flag")

    X, y = build_Xy_no_los(d, TASK6_NUM_FEATURES, TASK6_CAT_FEATURES, target_col="Breach_flag")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    preprocess = make_preprocess(TASK6_CAT_FEATURES)

    models: Dict[str, Pipeline] = {}

    # Logistic Regression
    lr = Pipeline(steps=[
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=3000))
    ])
    models["LogReg"] = lr

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
    )
    rf_pipe = Pipeline(steps=[("prep", preprocess), ("model", rf)])
    models["RF"] = rf_pipe

    # XGB optional
    if use_xgb and HAS_XGB:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / max(pos, 1)

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )
        xgb_pipe = Pipeline(steps=[("prep", preprocess), ("model", xgb)])
        models["XGB"] = xgb_pipe

    # Fit + probs on test
    y_prob: Dict[str, np.ndarray] = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_prob[name] = clf.predict_proba(X_test)[:, 1]

    return Task6TrainedBundle(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        models=models, y_prob=y_prob,
        seed=seed, test_size=test_size,
    )


# -----------------------
# Evaluation at a threshold (no retraining)
# -----------------------
def eval_at_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "cm00": int(cm[0, 0]), "cm01": int(cm[0, 1]),
        "cm10": int(cm[1, 0]), "cm11": int(cm[1, 1]),
    }


def evaluate_bundle(bundle: Task6TrainedBundle, threshold: float) -> pd.DataFrame:
    rows = []
    y_test = bundle.y_test
    for name, prob in bundle.y_prob.items():
        met = eval_at_threshold(y_test, prob, threshold)
        met["model"] = name
        met["n_train"] = int(len(bundle.y_train))
        met["n_test"] = int(len(bundle.y_test))
        rows.append(met)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


# -----------------------
# Plotting
# -----------------------
def fig_roc_models(bundle: Task6TrainedBundle, metrics_df: pd.DataFrame, threshold: float):
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=140)
    ax.plot([0, 1], [0, 1], linestyle="--")

    y_test = bundle.y_test
    for _, row in metrics_df.iterrows():
        name = row["model"]
        prob = bundle.y_prob[str(name)]
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={row['roc_auc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Task6 ROC — breach prediction (threshold={threshold})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_confusion_matrix_for_model(metrics_df: pd.DataFrame, model_name: str):
    r = metrics_df.loc[metrics_df["model"] == model_name].iloc[0]
    cm = np.array([[int(r["cm00"]), int(r["cm01"])],
                   [int(r["cm10"]), int(r["cm11"])]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")
    fig.tight_layout()
    return fig


# -----------------------
# Saving
# -----------------------
def save_task6_outputs(
    out_dir: Path,
    *,
    metrics_df: pd.DataFrame,
    fig_roc,
    figs_cm: Dict[str, object],
    tag: str,
    threshold: float,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"task6_metrics_{tag}_thr{threshold:.3f}.csv"
    roc_path = out_dir / f"task6_roc_{tag}_thr{threshold:.3f}.png"

    metrics_df.to_csv(metrics_path, index=False)
    fig_roc.savefig(roc_path, dpi=300, bbox_inches="tight")

    cm_paths = {}
    for name, fig in figs_cm.items():
        p = out_dir / f"task6_cm_{tag}_{name}_thr{threshold:.3f}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        cm_paths[name] = p

    paths = {"metrics_csv": metrics_path, "roc_png": roc_path}
    for name, p in cm_paths.items():
        paths[f"cm_png_{name}"] = p
    return paths
