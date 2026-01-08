# src/task5.py
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Reuse Task4 prep: coercion + Breach01 creation
from src.task4 import prepare_base


# -----------------------
# Defaults (adapt if your columns differ)
# -----------------------
DEFAULT_NUM_VARS = ["Age", "LoS", "noofpatients", "noofinvestigation", "nooftreatment"]
DEFAULT_CAT_VARS = ["DayofWeek", "Period", "HRG"]


def get_numeric_vars(df: pd.DataFrame) -> List[str]:
    """Return numeric candidates that exist in df."""
    return [c for c in DEFAULT_NUM_VARS if c in df.columns]


def get_cat_vars(df: pd.DataFrame) -> List[str]:
    """Return categorical candidates that exist in df."""
    return [c for c in DEFAULT_CAT_VARS if c in df.columns]


# -----------------------
# Core metrics
# -----------------------
def breach_rate_overall(df_s: pd.DataFrame) -> dict:
    """
    Compute overall breach rate using Breach01 created by prepare_base().
    Returns dict: n_valid, breach_count, breach_rate.
    """
    d = prepare_base(df_s)
    x = d["Breach01"].dropna()
    n = int(len(x))
    k = int((x == 1).sum())
    rate = float(k / n) if n > 0 else np.nan
    return {"n_valid": n, "breach": k, "breach_rate": rate}


def breach_rate_by_category(df_s: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """
    Group by categorical variable and compute:
    n, breach count, breach rate.
    """
    d = prepare_base(df_s)
    if cat_col not in d.columns:
        return pd.DataFrame()

    g = (
        d.dropna(subset=["Breach01"])
         .groupby(cat_col, dropna=False)["Breach01"]
         .agg(n="size", breach="sum", breach_rate="mean")
         .reset_index()
         .sort_values("breach_rate", ascending=False)
    )
    g["breach_rate"] = g["breach_rate"].astype(float)
    return g


def numeric_summary_by_breach(df_s: pd.DataFrame, num_col: str) -> pd.DataFrame:
    """
    Compare a numeric variable between Breach01=0 and Breach01=1.
    """
    d = prepare_base(df_s)
    if num_col not in d.columns:
        return pd.DataFrame()

    d[num_col] = pd.to_numeric(d[num_col], errors="coerce")

    out = (
        d.groupby("Breach01")[num_col]
         .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
         .reset_index()
    )
    # map breach label
    out["Breach01"] = out["Breach01"].map({0.0: "No breach", 1.0: "Breach", 0: "No breach", 1: "Breach"})
    return out


# -----------------------
# Optional helper: top categories by breach rate
# -----------------------
def top_categories_by_breach_rate(df_s: pd.DataFrame, cat_col: str, top_k: int = 10) -> pd.DataFrame:
    """
    Return top_k categories with highest breach rate (and minimum sample size filter).
    """
    tbl = breach_rate_by_category(df_s, cat_col)
    if tbl.empty:
        return tbl

    # Optional: avoid tiny categories misleading the top list
    tbl = tbl[tbl["n"] >= 5].copy()
    tbl = tbl.sort_values("breach_rate", ascending=False).head(top_k)
    return tbl.reset_index(drop=True)
