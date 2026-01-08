# src/task8_data.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# ---------------------------
# Logging
# ---------------------------
LOG_PATH = Path("outputs") / "logs" / "app.log"

def log_event(
    action: str,
    success: bool,
    details: dict | None = None,
    error: str | None = None,
) -> None:
    """Append one JSON line to outputs/logs/app.log"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    rec = {
        "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "success": bool(success),
        "details": details or {},
        "error": error or "",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------
# CRUD helpers
# ---------------------------
def retrieve_by_id(df: pd.DataFrame, id_col: str, value: str) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found.")
    if value is None or str(value).strip() == "":
        raise ValueError("Patient ID is empty.")
    v = str(value).strip()
    return df[df[id_col].astype(str) == v].copy()


def filter_range(df: pd.DataFrame, col: str, vmin: float, vmax: float) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    if vmin > vmax:
        raise ValueError("Min cannot be > Max.")
    s = pd.to_numeric(df[col], errors="coerce")
    mask = (s >= vmin) & (s <= vmax)
    return df.loc[mask].copy()


def delete_by_patient_id(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str = "ID",
) -> tuple[pd.DataFrame, int]:
    """Delete rows matching patient_id. Returns (new_df, deleted_count)."""
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found.")

    pid = str(patient_id).strip()
    if pid == "":
        raise ValueError("patient_id is empty.")

    mask = df[id_col].astype(str) == pid
    deleted = int(mask.sum())
    if deleted == 0:
        raise ValueError(f"No rows found for patient ID '{pid}'.")

    new_df = df.loc[~mask].copy()
    return new_df, deleted


def modify_value_by_patient_id(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str,
    col: str,
    new_value,
) -> tuple[pd.DataFrame, list]:
    """Modify value for all rows matching patient_id. Returns (new_df, old_values_list)."""
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found.")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    pid = str(patient_id).strip()
    if pid == "":
        raise ValueError("patient_id is empty.")

    out = df.copy()
    mask = out[id_col].astype(str) == pid
    affected = int(mask.sum())
    if affected == 0:
        raise ValueError(f"No rows found for patient_id={pid}")

    old_vals = out.loc[mask, col].tolist()

    if pd.api.types.is_numeric_dtype(out[col]):
        new_value = pd.to_numeric(new_value, errors="raise")

    out.loc[mask, col] = new_value
    return out, old_vals
