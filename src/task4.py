# src/task4.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Set

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap


# -----------------------
# Task 4.1 constants
# -----------------------
HRG_TOP_N = 10

CROWDING_BIN_METHOD = "quantile3"  # or "fixed"
FIXED_CROWDING_BINS = [0, 10, 20, np.inf]
FIXED_CROWDING_LABELS = ["Low (0–9)", "Medium (10–19)", "High (20+)"]


# -----------------------
# Task 4.1 helper funcs
# -----------------------
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def coerce_breach_to_01(s: pd.Series) -> pd.Series:
    """Ensure breach is numeric 0/1. Handles 0/1, Yes/No, True/False, Breach/No breach."""
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        return out.where(out.isin([0, 1]))
    v = s.astype(str).str.strip().str.lower()
    yes = {"1", "yes", "true", "breach", "breached"}
    no = {"0", "no", "false", "non-breach", "nonbreach", "no breach", "nobreach"}
    mapped = np.where(v.isin(yes), 1, np.where(v.isin(no), 0, np.nan))
    return pd.Series(mapped, index=s.index, name=s.name, dtype="float")


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preparation:
    - coerce numeric columns
    - create Breach01 (0/1)
    """
    d = df.copy()
    coerce_numeric(
        d,
        ["Age", "Day", "Period", "LoS",
         "noofinvestigation", "nooftreatment", "noofpatients"]
    )

    if "Breachornot" not in d.columns:
        raise KeyError("Column 'Breachornot' not found. (Check standardize_columns_to_sample)")

    d["Breach01"] = coerce_breach_to_01(d["Breachornot"])
    return d


def add_crowding_bins_3level(df: pd.DataFrame) -> pd.DataFrame:
    """Low/Medium/High crowding bins based on noofpatients."""
    if "noofpatients" not in df.columns:
        raise KeyError("Column 'noofpatients' not found.")
    s = df.copy()
    s["noofpatients"] = pd.to_numeric(s["noofpatients"], errors="coerce")

    if CROWDING_BIN_METHOD == "fixed":
        s["crowding_bin3"] = pd.cut(
            s["noofpatients"],
            bins=FIXED_CROWDING_BINS,
            labels=FIXED_CROWDING_LABELS,
            right=False,
            include_lowest=True,
        )
    else:
        s["crowding_bin3"] = pd.qcut(
            s["noofpatients"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )
    return s


def task4_summary_table(
    df: pd.DataFrame,
    out_dir: Path,
    numeric_vars: Optional[List[str]] = None,
    out_name: str = "Task4_summary_table.xlsx",
    *,
    auto_detect_numeric: bool = True,
    exclude_vars: Optional[Set[str]] = None,
    include_quartiles: bool = True,
    coerce_numeric_cols: bool = True,
    save_csv: bool = True,
    treat_as_categorical: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Summary statistics table for numeric variables.
    Saves to out_dir/out_name (+ optional CSV).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if exclude_vars is None:
        exclude_vars = {"ID"}

    if treat_as_categorical is None:
        treat_as_categorical = {"DayofWeek", "Period", "Day", "HRG", "Breachornot"}

    d = df.copy()

    # 1) decide numeric vars
    if numeric_vars is None:
        if not auto_detect_numeric:
            numeric_vars = ["Age", "LoS", "noofpatients", "noofinvestigation", "nooftreatment"]
        else:
            numeric_vars = [
                c for c in d.columns
                if (c not in exclude_vars)
                and (c not in treat_as_categorical)
                and pd.api.types.is_numeric_dtype(d[c])
            ]

            # object candidates that are mostly numeric
            obj_cands = [
                c for c in d.columns
                if (c not in exclude_vars)
                and (c not in treat_as_categorical)
                and (d[c].dtype == "object")
            ]
            for c in obj_cands:
                tmp = pd.to_numeric(d[c], errors="coerce")
                if tmp.notna().mean() >= 0.8:
                    d[c] = tmp
                    numeric_vars.append(c)

            numeric_vars = sorted(set(numeric_vars), key=lambda x: list(d.columns).index(x))
    else:
        numeric_vars = [
            c for c in numeric_vars
            if c in d.columns and c not in exclude_vars and c not in treat_as_categorical
        ]

    if not numeric_vars:
        raise ValueError("No numeric variables found for summary table. Check column names/types.")

    # 2) coerce chosen numeric vars
    if coerce_numeric_cols:
        for c in numeric_vars:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # 3) summary
    summary = pd.DataFrame(index=numeric_vars)
    summary["Mean"] = d[numeric_vars].mean()
    summary["Median"] = d[numeric_vars].median()
    summary["Std"] = d[numeric_vars].std()
    summary["Min"] = d[numeric_vars].min()
    if include_quartiles:
        summary["P25"] = d[numeric_vars].quantile(0.25)
        summary["P75"] = d[numeric_vars].quantile(0.75)
        summary["IQR"] = summary["P75"] - summary["P25"]
    summary["Max"] = d[numeric_vars].max()
    summary["Missing"] = d[numeric_vars].isna().sum()
    summary["N (non-missing)"] = d[numeric_vars].notna().sum()

    summary = summary.round(2)

    out_path = out_dir / out_name
    summary.to_excel(out_path)
    if save_csv:
        summary.to_csv(out_path.with_suffix(".csv"))

    return summary


# -----------------------
# Task 4.1 plot funcs
# -----------------------
def plot_age_composite(
    df: pd.DataFrame,
    out_dir: Path,
    out_name: str = "task4_age_composite_figure.png",
    show: bool = False,
) -> Path:
    """Age distribution & age group composition (original styling preserved)."""
    from scipy.stats import gaussian_kde

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    d = df.copy()
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    age = d["Age"].dropna().to_numpy()

    age_bins = [0, 17, 34, 49, 64, 120]
    age_labels = ["0–17", "18–34", "35–49", "50–64", "65+"]

    d["Age_group"] = pd.cut(d["Age"], bins=age_bins, labels=age_labels, include_lowest=True)
    age_counts = d["Age_group"].value_counts().reindex(age_labels).fillna(0)
    age_pct = (age_counts / age_counts.sum() * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    ax.hist(age, bins=25, density=True, color="#9AF5DB", edgecolor="white", alpha=0.6)
    kde = gaussian_kde(age)
    x = np.linspace(age.min(), age.max(), 400)
    ax.plot(x, kde(x), color="#2f7f8a", linewidth=2.5)
    ax.set_title("A. Age distribution", fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    colors = ["#d7f0f0", "#bfe3e5", "#93c9cf", "#5aa7b0", "#2f7f8a"]
    bars = ax.bar(age_labels, age_counts.values, color=colors, edgecolor="white", linewidth=1)
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(age_counts) * 0.02,
            f"{int(age_counts.iloc[i])}\n({age_pct.iloc[i]}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("B. Age group composition", fontweight="bold")
    ax.set_ylabel("Number of patients")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.suptitle(f"Age characteristics of AED sample (n={len(d)})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_hrg_distribution(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hrg_counts = df["HRG"].value_counts().head(top_n).sort_values()
    total = len(df)
    percentages = hrg_counts / total * 100

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cmap = plt.get_cmap("YlGnBu")
    colors = cmap(np.linspace(0.35, 0.85, len(hrg_counts)))

    bars = ax.barh(hrg_counts.index, hrg_counts.values, color=colors, edgecolor="none")
    for bar, count, pct in zip(bars, hrg_counts.values, percentages.values):
        ax.text(count + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{count} ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_xlabel("Number of patients", fontsize=11)
    ax.set_ylabel("HRG category", fontsize=11)
    ax.set_title(f"Top {top_n} HRG categories in AED sample (n={total})",
                 fontsize=13, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_crowding_task4(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    crowd = pd.to_numeric(df["noofpatients"], errors="coerce").dropna()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    by_day = df.groupby("DayofWeek")["noofpatients"].mean().reindex(weekday_order)
    counts = df["DayofWeek"].value_counts().reindex(weekday_order)
    percentages = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.1, 1.4]})

    ax1 = axes[0]
    ax1.hist(crowd, bins=25, color="#E69AF5", alpha=0.85, edgecolor="white")
    ax1.set_title("A. Distribution of crowding on arrival", loc="left", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of patients on arrival")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    ax2 = axes[1]
    bars = ax2.bar(weekday_order, by_day.values, color="#E69AF5", alpha=0.8, edgecolor="white")
    ax2.set_title("B. Crowding on arrival by day of week", loc="left", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Day of week")
    ax2.set_ylabel("Mean number of patients on arrival")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    ax2b = ax2.twinx()
    ax2b.plot(weekday_order, percentages.values, color="#6FDDED", marker="o", linewidth=2)
    ax2b.set_ylabel("Percentage of weekly arrivals (%)")

    for bar, pct in zip(bars, percentages.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Crowding characteristics of AED sample (n = 400)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_los_and_investigation(df: pd.DataFrame, out_path: Path,
                               los_col: str = "LoS", inv_col: str = "noofinvestigation") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = df[[los_col, inv_col]].copy()
    d[los_col] = pd.to_numeric(d[los_col], errors="coerce")
    d[inv_col] = pd.to_numeric(d[inv_col], errors="coerce")
    d = d.dropna()

    p99 = d[los_col].quantile(0.99)
    d_main = d[d[los_col] <= p99]

    d["inv_group"] = d[inv_col].clip(upper=5)
    order = sorted(d["inv_group"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Length of Stay characteristics of AED sample (n = 400)",
                 fontsize=16, fontweight="bold", y=1.03)

    ax1.hist(d_main[los_col], bins=22, density=True, color="#84F5C0", edgecolor="white", alpha=0.85)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(d_main[los_col])
        x = np.linspace(d_main[los_col].min(), d_main[los_col].max(), 300)
        ax1.plot(x, kde(x), color="#1F6F63", linewidth=2)
    except Exception:
        pass

    ax1.set_title("A. Length of stay distribution (main body)", loc="left", fontweight="bold")
    ax1.set_xlabel("Length of stay (LoS)")
    ax1.set_ylabel("Density")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.text(0.98, 0.95, f"Histogram capped at P99 = {int(p99)}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85"))

    data = [d.loc[d["inv_group"] == g, los_col] for g in order]
    ax2.boxplot(
        data, patch_artist=True, widths=0.6, showfliers=True,
        boxprops=dict(facecolor="#84F5C0", color="#1F6F63", alpha=0.6),
        medianprops=dict(color="#1F6F63", linewidth=2),
        whiskerprops=dict(color="#1F6F63"),
        capprops=dict(color="#1F6F63"),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="white",
                        markeredgecolor="#1F6F63", alpha=0.8),
    )

    ax2.set_xticks(range(1, len(order) + 1))
    ax2.set_xticklabels([str(int(g)) if g < 5 else "5+" for g in order])
    ax2.set_xlabel("Number of investigation")
    ax2.set_ylabel("Length of stay (LoS)")
    ax2.set_title("B. Length of stay by investigation intensity", loc="left", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_breach_vs_crowding(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    los_col = "LoS"
    breach_col = "Breachornot"
    crowd_col = "noofpatients"

    d = df[[los_col, breach_col, crowd_col]].copy()
    d[los_col] = pd.to_numeric(d[los_col], errors="coerce")
    d[crowd_col] = pd.to_numeric(d[crowd_col], errors="coerce")
    d["breach01"] = coerce_breach_to_01(d[breach_col])

    d = d.dropna(subset=[los_col, crowd_col, "breach01"]).copy()
    d["breach01"] = d["breach01"].astype(int)
    d["breach_label"] = np.where(d["breach01"] == 1, "Breach", "Not breach")

    d["crowd_q"] = pd.qcut(d[crowd_col], q=4, duplicates="drop")
    qcats = d["crowd_q"].cat.categories
    labels = [f"Q{i+1}\n[{int(c.left)}–{int(c.right)}]" for i, c in enumerate(qcats)]
    d["crowd_lbl"] = d["crowd_q"].cat.rename_categories(labels)

    grp = d.groupby("crowd_lbl", observed=True).agg(
        n=("breach01", "size"),
        breach_rate=("breach01", "mean"),
    ).reset_index()

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("grad", ["#E9F024", "#D69BF2", "#439FF0"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.2]})
    fig.suptitle(f"Breach risk and crowding proxy (n = {len(d)})",
                 fontsize=16, fontweight="bold", y=1.03)

    cols = [cmap(x) for x in np.linspace(0.15, 0.85, len(grp))]
    ax1.bar(grp["crowd_lbl"], grp["breach_rate"], color=cols, edgecolor="white", alpha=0.95)
    ax1.set_title("A. Breach rate by crowding level (arrival)", loc="left", fontweight="bold")
    ax1.set_xlabel("Crowding quartile (no. of patients already in AED)")
    ax1.set_ylabel("Breach rate")
    ax1.set_ylim(0, max(0.05, grp["breach_rate"].max() * 1.25))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    for i, (rate, n) in enumerate(zip(grp["breach_rate"], grp["n"])):
        ax1.text(i, rate, f"n={int(n)}", ha="center", va="bottom", fontsize=10)

    order = ["Not breach", "Breach"]
    data = [d.loc[d["breach_label"] == g, los_col] for g in order]
    bp = ax2.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="0.25"), capprops=dict(color="0.25"),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="white",
                        markeredgecolor="0.25", alpha=0.75),
    )
    colors2 = [cmap(0.2), cmap(0.8)]
    for patch, c in zip(bp["boxes"], colors2):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("0.25")

    ax2.set_xticklabels(order)
    ax2.set_title("B. Length of stay by breach status", loc="left", fontweight="bold")
    ax2.set_ylabel("Length of stay (LoS, minutes)")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Task 4.1 entry (MUST be last)
# -----------------------
def run_task4_1(df_s: pd.DataFrame, out_dir: Path) -> None:
    """
    Task 4.1: main features of variables (summary + several figures)
    """
    df = prepare_base(df_s)

    # Summary table
    task4_summary_table(df, out_dir=out_dir, out_name="Task4_summary_table.xlsx")

    # Age (A+B)
    plot_age_composite(
        df,
        out_dir=out_dir,
        out_name="task4_age_composite_figure.png",
        show=False
    )

    # HRG
    plot_hrg_distribution(df, out_path=out_dir / "task4_hrg_top10.png", top_n=HRG_TOP_N)

    # Crowding
    plot_crowding_task4(df, out_path=out_dir / "task4_crowding_overview.png")

    # LoS vs investigation
    plot_los_and_investigation(df, out_path=out_dir / "task4_los_vs_investigation.png")

    # Breach vs crowding
    plot_breach_vs_crowding(df, out_path=out_dir / "task4_breach_vs_crowding.png")

# =======================
# Task 4.2 Relationships
# =======================




# ---------- small helpers ----------

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def _breach_to_01(s: pd.Series) -> pd.Series:
    """Robust breach -> 0/1 (works for 'breach'/'non-breach', yes/no, 0/1, true/false)."""
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        return (x > 0).astype("Int64")

    x = s.astype(str).str.strip().str.lower()
    yes = {"1", "true", "yes", "y", "breach", "breached"}
    no = {"0", "false", "no", "n", "non-breach", "nonbreach", "no breach", "nobreach"}
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[x.isin(yes)] = 1
    out[x.isin(no)] = 0
    # fallback contains
    out[out.isna() & x.str.contains("breach", na=False)] = 1
    out[out.isna() & x.str.contains("non", na=False)] = 0
    return out.astype("Int64")


def _gradient_line(ax, x, y, cvals, cmap, lw=3.0, alpha=0.98, zorder=6):
    """Draw a line with color gradient along its length."""
    x = np.asarray(x); y = np.asarray(y); cvals = np.asarray(cvals)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(cvals[:-1])
    ax.add_collection(lc)
    return lc


def _binned_trend(x, y, bins=6):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return None, None
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    xc, yc = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (x >= lo) & (x <= hi)
        if mask.sum() >= 5:
            xc.append(np.median(x[mask]))
            yc.append(np.mean(y[mask]))
    return np.array(xc), np.array(yc)


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score CI for a proportion. Return (p, lo, hi)."""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2*n)) / denom
    half = (z / denom) * np.sqrt((phat*(1-phat) + z**2/(4*n)) / n)
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (phat, lo, hi)


# ---------- 4.2 Q2: temporal crowding trend ----------

def plot_day_trend_crowding(df_s: pd.DataFrame, out_path: Path, roll_win: int = 7) -> None:
    DAY_COL = "Day"
    CROWD_COL = "noofpatients"
    GRAD_COLORS = ["#2C7FB8", "#41B6C4", "#7FCDBB", "#D9F0A3", "#E9F507"]

    d = df_s[[DAY_COL, CROWD_COL]].copy()
    d[DAY_COL] = pd.to_numeric(d[DAY_COL], errors="coerce")
    d[CROWD_COL] = pd.to_numeric(d[CROWD_COL], errors="coerce")
    d = d.dropna(subset=[DAY_COL, CROWD_COL])

    d["day_int"] = d[DAY_COL].round().astype(int)

    g = (
        d.groupby("day_int", as_index=False)
         .agg(mean_crowd=(CROWD_COL, "mean"),
              std_crowd=(CROWD_COL, "std"),
              n=(CROWD_COL, "size"))
         .sort_values("day_int")
    )

    g["se"] = g["std_crowd"] / np.sqrt(g["n"].clip(lower=1))
    g["roll_mean"] = g["mean_crowd"].rolling(roll_win, min_periods=2, center=True).mean()
    g["roll_mean"] = g["roll_mean"].fillna(g["mean_crowd"])

    cmap = LinearSegmentedColormap.from_list("blue_green_yellow", GRAD_COLORS)
    norm = Normalize(vmin=g["roll_mean"].min(), vmax=g["roll_mean"].max())
    cvals = norm(g["roll_mean"].values)

    norm_n = Normalize(vmin=g["n"].min(), vmax=g["n"].max())
    bar_colors = cmap(norm_n(g["n"].values))

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    fig = plt.figure(figsize=(14.8, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    fig.suptitle(
        f"Temporal pattern of crowding over 4 weeks (n = {len(d)})",
        fontsize=16, fontweight="bold", y=0.98
    )

    sc = ax1.scatter(
        g["day_int"].values,
        g["mean_crowd"].values,
        c=g["roll_mean"].values,
        cmap=cmap,
        s=48,
        edgecolor="white",
        linewidth=0.9,
        zorder=5
    )

    _gradient_line(
        ax1,
        g["day_int"].values,
        g["roll_mean"].values,
        cvals=cvals,
        cmap=cmap,
        lw=3.2,
        alpha=0.98,
        zorder=6
    )

    ax1.set_title("A. Daily crowding level (mean) with 7-day rolling trend",
                  loc="left", fontweight="bold")
    ax1.set_ylabel("Crowding proxy: number of patients on arrival")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)

    cbar = fig.colorbar(sc, ax=ax1, pad=0.01, fraction=0.045)
    cbar.set_label("Rolling mean (colour scale)")

    ax2.bar(
        g["day_int"].values,
        g["n"].values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95
    )

    ax2.set_title("B. Daily sample size (n) in the selected AED sample",
                  loc="left", fontweight="bold")
    ax2.set_xlabel("Day index (within 28-day window)")
    ax2.set_ylabel("n (arrivals)")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_axisbelow(True)

    xmin, xmax = int(g["day_int"].min()), int(g["day_int"].max())
    ax2.set_xticks(list(range(xmin, xmax + 1, 2)))

    ax2.text(0.01, -0.28,
             f"Notes: Rolling window = {roll_win} days.",
             transform=ax2.transAxes, ha="left", va="top",
             fontsize=10, color="0.25")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- 4.2 Q3: LoS vs breach/patients/investigation ----------

def plot_interesting_relationships_los(df_s: pd.DataFrame, out_path: Path, seed: int) -> None:
    COL_NOBREACH = "#57F7DA"
    COL_BREACH   = "#57ACF7"
    COL_LINE     = "#57D7F7"
    COL_GRID     = "#EAF2F8"
    COL_TEXT     = "#1F2937"

    mpl.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})

    COL_LOS = "LoS"
    COL_PAT = "noofpatients"
    COL_INV = "noofinvestigation"
    COL_BR  = "Breachornot"

    s = df_s.copy()
    s[COL_LOS] = _to_num(s[COL_LOS])
    s[COL_PAT] = _to_num(s[COL_PAT])
    s[COL_INV] = _to_num(s[COL_INV])
    s["breach"] = _breach_to_01(s[COL_BR])

    s = s.dropna(subset=[COL_LOS, "breach"])
    s["breach"] = s["breach"].astype(int)

    los_b  = s.loc[s["breach"] == 1, COL_LOS].values
    los_nb = s.loc[s["breach"] == 0, COL_LOS].values

    fig = plt.figure(figsize=(10.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.25, wspace=0.25)

    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    fig.suptitle("Task 4.2 — Interesting relationships (sample n = 400)", y=0.98)

    bp = axA.boxplot(
        [los_b, los_nb],
        positions=[1, 2],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COL_TEXT, linewidth=1.8),
        boxprops=dict(linewidth=1.2, color=COL_TEXT),
        whiskerprops=dict(color=COL_TEXT),
        capprops=dict(color=COL_TEXT),
    )
    bp["boxes"][0].set_facecolor(COL_BREACH)
    bp["boxes"][1].set_facecolor(COL_NOBREACH)
    bp["boxes"][0].set_alpha(0.45)
    bp["boxes"][1].set_alpha(0.45)

    rng = np.random.default_rng(seed)
    axA.scatter(rng.normal(1, 0.06, len(los_b)), los_b, s=18, color=COL_BREACH, alpha=0.5)
    axA.scatter(rng.normal(2, 0.06, len(los_nb)), los_nb, s=18, color=COL_NOBREACH, alpha=0.35)

    axA.set_xticks([1, 2])
    axA.set_xticklabels([f"Breach (n={len(los_b)})", f"No breach (n={len(los_nb)})"])
    axA.set_ylabel("Length of Stay (minutes)")
    axA.set_title("A) Length of Stay by breach status", pad=8)

    axA.text(0.99, 0.02,
             f"Median LoS: breach={np.median(los_b):.0f} min | no breach={np.median(los_nb):.0f} min",
             transform=axA.transAxes, ha="right", va="bottom")

    axA.grid(axis="y", color=COL_GRID)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    for lab, mask, col in [
        ("No breach", s["breach"] == 0, COL_NOBREACH),
        ("Breach",    s["breach"] == 1, COL_BREACH),
    ]:
        x = s.loc[mask, COL_PAT].values
        y = s.loc[mask, COL_LOS].values
        axB.scatter(x, y, s=18, color=col, alpha=0.45, label=lab)
        xc, yc = _binned_trend(x, y)
        if xc is not None:
            axB.plot(xc, yc, color=COL_LINE, linewidth=2.2)

    axB.set_title("B) LoS vs patients already in AED on arrival", pad=6)
    axB.set_xlabel("Patients on arrival")
    axB.set_ylabel("LoS (minutes)")
    axB.grid(color=COL_GRID)
    axB.legend(frameon=False)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    for lab, mask, col in [
        ("No breach", s["breach"] == 0, COL_NOBREACH),
        ("Breach",    s["breach"] == 1, COL_BREACH),
    ]:
        x = s.loc[mask, COL_INV].values
        y = s.loc[mask, COL_LOS].values
        axC.scatter(x, y, s=18, color=col, alpha=0.45)
        xc, yc = _binned_trend(x, y)
        if xc is not None:
            axC.plot(xc, yc, color=COL_LINE, linewidth=2.2)

    axC.set_title("C) LoS vs process intensity (investigation)", pad=6)
    axC.set_xlabel("Number of investigation")
    axC.set_ylabel("LoS (minutes)")
    axC.grid(color=COL_GRID)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


# ---------- 4.2 Q4: day-of-week breach pattern ----------

def plot_workload_breach_by_dayofweek(df_s: pd.DataFrame, out_path: Path) -> None:
    DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    DARK   = "#3D246C"
    MID    = "#7B4CCB"
    NB_COL = "#F3C6E6"
    B_COL  = "#6C43B5"
    BOX_F1 = "#E9DBFF"
    BOX_F2 = "#CBB7FF"
    GRID   = "#D8D1E8"

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    sample = df_s.copy()
    sample["DayofWeek"] = pd.Categorical(sample["DayofWeek"], categories=DAY_ORDER, ordered=True)
    sample["noofpatients"] = pd.to_numeric(sample["noofpatients"], errors="coerce")
    sample["Breach_bin"] = _breach_to_01(sample["Breachornot"]).astype("Int64")

    g = sample.groupby("DayofWeek", observed=True)

    dist_by_day = [
        g.get_group(day)["noofpatients"].dropna().values if day in g.groups else np.array([])
        for day in DAY_ORDER
    ]

    counts = g["Breach_bin"].agg(["size", "sum"]).reindex(DAY_ORDER)
    counts.columns = ["n", "breach"]
    counts["non_breach"] = counts["n"] - counts["breach"]
    counts["breach_rate"] = np.where(counts["n"] > 0, counts["breach"] / counts["n"], np.nan)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.2, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.18}
    )

    fig.suptitle("Workload and breach pattern by day of week (sample n = 400)",
                 y=0.98, color=DARK)

    bp = ax1.boxplot(
        dist_by_day,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops=dict(color=DARK, linewidth=1.7),
        whiskerprops=dict(color=DARK, linewidth=1.1),
        capprops=dict(color=DARK, linewidth=1.1),
        boxprops=dict(edgecolor=DARK, linewidth=1.1),
    )

    fills = []
    for i in range(len(DAY_ORDER)):
        t = i / max(1, (len(DAY_ORDER) - 1))
        c = (1 - t) * np.array(mpl.colors.to_rgb(BOX_F1)) + t * np.array(mpl.colors.to_rgb(BOX_F2))
        fills.append(mpl.colors.to_hex(c))

    for box, c in zip(bp["boxes"], fills):
        box.set_facecolor(c)
        box.set_alpha(0.95)

    means = [np.mean(x) if len(x) else np.nan for x in dist_by_day]
    ax1.scatter(np.arange(1, len(DAY_ORDER) + 1), means, s=18, color=DARK, alpha=0.75, zorder=3)

    ax1.set_ylabel("Patients already in AED on arrival\n(noofpatients)")
    ax1.grid(axis="y", color=GRID, alpha=0.35, linewidth=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    x = np.arange(len(DAY_ORDER))
    bar_w = 0.72

    ax2.bar(x, counts["non_breach"].values, width=bar_w, color=NB_COL, edgecolor="none",
            label="No breach (count)")
    ax2.bar(x, counts["breach"].values, width=bar_w, bottom=counts["non_breach"].values,
            color=B_COL, edgecolor="none", label="Breach (count)")

    ax2.set_ylabel("Patient count")
    ax2.grid(axis="y", color=GRID, alpha=0.35, linewidth=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2r = ax2.twinx()
    rate = counts["breach_rate"].values * 100
    ax2r.plot(x, rate, color=MID, linewidth=1.9, marker="o", markersize=6,
              markerfacecolor=MID, markeredgecolor=MID, zorder=5)
    ax2r.set_ylabel("Breach rate (%)")
    ax2r.spines["top"].set_visible(False)

    ymax = np.nanmax(rate)
    ax2r.set_ylim(0, max(4.2, ymax + 1.1))

    for i, r in enumerate(rate):
        if np.isfinite(r):
            ax2r.text(i, r + 0.22, f"{r:.1f}%",
                      ha="center", va="bottom", fontsize=9, color=DARK,
                      bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                edgecolor="none", alpha=0.65),
                      zorder=6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(DAY_ORDER)
    ax2.set_xlabel("Day of week")

    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.12),
               ncol=2, columnspacing=1.4, handlelength=1.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- 4.2 Q4: arrival period relationships (LoS vs workload/inv/trt) ----------

def plot_arrival_period_relationships(df_s: pd.DataFrame, out_path: Path) -> None:
    PERIOD_ORDER = list(range(24))

    CMAP   = mpl.cm.YlGnBu
    ACCENT = "#97FCEF"
    DARK   = "#1F2D3A"
    GRID   = "#E9F1F5"

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    sample = df_s.copy()

    COL_PERIOD = "Period"
    COL_LOS = "LoS"
    COL_PAT = "noofpatients"
    COL_INV = "noofinvestigation"
    COL_TRT = "nooftreatment"

    for c in [COL_PERIOD, COL_LOS, COL_PAT, COL_INV, COL_TRT]:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")

    sample = sample[sample[COL_PERIOD].isin(PERIOD_ORDER)]
    sample[COL_PERIOD] = pd.Categorical(sample[COL_PERIOD], categories=PERIOD_ORDER, ordered=True)

    g = sample.groupby(COL_PERIOD, observed=True)

    los_by_period = [
        g.get_group(p)[COL_LOS].dropna().values if p in g.groups else np.array([])
        for p in PERIOD_ORDER
    ]

    summary = g.agg(
        n=(COL_LOS, "size"),
        los_mean=(COL_LOS, "mean"),
        pat_mean=(COL_PAT, "mean"),
        inv_mean=(COL_INV, "mean"),
        trt_mean=(COL_TRT, "mean"),
    ).reindex(PERIOD_ORDER)

    fig = plt.figure(figsize=(11.6, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.22)

    ax_top = fig.add_subplot(gs[0, 0])
    gs_bot = gs[1, 0].subgridspec(1, 3, wspace=0.30)
    ax1, ax2, ax3 = [fig.add_subplot(gs_bot[0, i]) for i in range(3)]

    fig.suptitle(
        "Arrival period relationships: workload / investigation / treatment vs LoS (sample n = 400)",
        y=0.97, color=DARK
    )

    bp = ax_top.boxplot(
        los_by_period,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops=dict(color=DARK, linewidth=1.7),
        whiskerprops=dict(color=DARK, linewidth=1.15),
        capprops=dict(color=DARK, linewidth=1.15),
        boxprops=dict(edgecolor=DARK, linewidth=1.0),
    )

    fills = [CMAP(v) for v in np.linspace(0.45, 0.98, 24)]
    for box, c in zip(bp["boxes"], fills):
        box.set_facecolor(c)
        box.set_alpha(0.98)

    means = [np.mean(x) if len(x) else np.nan for x in los_by_period]
    ax_top.scatter(np.arange(1, 25), means, s=22, color=DARK, alpha=0.75, zorder=3)

    ax_top.set_title("LoS distribution by arrival period", pad=10)
    ax_top.set_ylabel("Length of Stay (minutes)")
    ax_top.grid(axis="y", color=GRID, alpha=0.40, linewidth=0.6)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_xticks(np.arange(1, 25))
    ax_top.set_xticklabels([])

    periods = np.array(PERIOD_ORDER)
    norm = mpl.colors.Normalize(0, 23)
    colors = CMAP(norm(periods))
    sizes = 40 + 200 * (summary["n"].values / summary["n"].max())

    def add_reg(ax, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            b, a = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(x[m].min(), x[m].max(), 100)
            ax.plot(xx, a + b * xx, color=ACCENT, lw=2.8)

    def style(ax, xlabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean LoS (min)")
        ax.grid(True, color=GRID, alpha=0.18, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.scatter(summary["pat_mean"], summary["los_mean"], s=sizes, c=colors,
                alpha=0.95, edgecolor="white", linewidth=0.6)
    add_reg(ax1, summary["pat_mean"].values, summary["los_mean"].values)
    style(ax1, "Mean patients on arrival")

    ax2.scatter(summary["inv_mean"], summary["los_mean"], s=sizes, c=colors,
                alpha=0.95, edgecolor="white", linewidth=0.6)
    add_reg(ax2, summary["inv_mean"].values, summary["los_mean"].values)
    style(ax2, "Mean investigation")

    ax3.scatter(summary["trt_mean"], summary["los_mean"], s=sizes, c=colors,
                alpha=0.95, edgecolor="white", linewidth=0.6)
    add_reg(ax3, summary["trt_mean"].values, summary["los_mean"].values)
    style(ax3, "Mean treatment")

    cax = fig.add_axes([0.92, 0.16, 0.015, 0.28])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=norm)
    cb.set_label("Arrival period (0–23)")
    cb.outline.set_visible(False)

    fig.text(
        0.12, 0.06,
        "Note: each point represents one arrival period; point size reflects sample size in that period.",
        fontsize=10, color=DARK
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(bottom=0.12, right=0.9)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

#=======================================================================
# ---------- 4.2 Q4: breach by arrival period (rate + counts) ----------

def plot_breach_by_period(df_s: pd.DataFrame, out_path: Path) -> None:
    RATE_BAR = "#36EBB8"
    NO_BREACH = "#36EBEB"
    BREACH = "#F7F257"
    DARK = "#0B1320"
    GRID = "#E7EDF4"

    MIN_N_FOR_CI = 20
    MIN_BREACH_FOR_CI = 2

    sample = df_s.copy()
    sample["Period"] = pd.to_numeric(sample["Period"], errors="coerce").astype("Int64")
    sample["Breach_bin"] = _breach_to_01(sample["Breachornot"]).astype("Int64")

    hours = np.arange(24)
    g = sample.dropna(subset=["Period"]).groupby("Period")["Breach_bin"]
    tbl = g.agg(n="size", breach="sum").reindex(hours, fill_value=0)
    tbl["non_breach"] = tbl["n"] - tbl["breach"]

    p = np.full(24, np.nan)
    lo = np.full(24, np.nan)
    hi = np.full(24, np.nan)

    for h in hours:
        p[h], lo[h], hi[h] = _wilson_ci(int(tbl.loc[h, "breach"]), int(tbl.loc[h, "n"]))

    rate = p * 100
    err_lo = np.clip((p - lo) * 100, 0, None)
    err_hi = np.clip((hi - p) * 100, 0, None)

    mask_rate = tbl["breach"].values > 0
    mask_ci = (
        (tbl["n"].values >= MIN_N_FOR_CI) &
        (tbl["breach"].values >= MIN_BREACH_FOR_CI) &
        np.isfinite(rate)
    )

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(11.6, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.08}
    )

    fig.suptitle("Breach pattern by arrival period (sample n = 400)",
                 y=0.98, fontsize=13, color=DARK)

    x = hours

    axA.bar(x[mask_rate], rate[mask_rate], color=RATE_BAR, width=0.72, edgecolor="none", alpha=0.95)
    axA.errorbar(x[mask_ci], rate[mask_ci],
                 yerr=[err_lo[mask_ci], err_hi[mask_ci]],
                 fmt="none", ecolor=DARK, elinewidth=0.9, capsize=2.5, alpha=0.55)

    for i in x[mask_ci]:
        axA.text(i, 0.35, f"n={int(tbl.loc[i,'n'])}",
                 ha="center", va="bottom", fontsize=9, color=DARK)

    axA.set_ylabel("Breach rate (%)")
    axA.set_title("Breach rate (Wilson 95% CI shown only for informative periods)", fontsize=11)
    axA.grid(axis="y", color=GRID, alpha=0.5, linewidth=0.6)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    axB.bar(x, tbl["non_breach"], color=NO_BREACH, width=0.72, edgecolor="none",
            label="No breach (count)")
    axB.bar(x, tbl["breach"], bottom=tbl["non_breach"], color=BREACH, width=0.72,
            edgecolor="none", label="Breach (count)")

    axB.set_ylabel("Patient count")
    axB.set_xlabel("Arrival period (hour of day)")
    axB.legend(frameon=False, ncol=2, loc="upper left")
    axB.grid(axis="y", color=GRID, alpha=0.5, linewidth=0.6)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    axB.set_xticks(x)
    axB.set_xticklabels([f"{h:02d}" for h in x])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- 4.2 Q5: association heatmap ----------

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    df = pd.crosstab(x, y)
    if df.size == 0:
        return np.nan
    observed = df.to_numpy()
    n = observed.sum()
    if n == 0:
        return np.nan

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n

    mask = expected > 0
    chi2 = ((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum()

    r, k = observed.shape
    if r <= 1 or k <= 1:
        return 0.0

    phi2 = chi2 / n
    if n > 1:
        phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        kcorr = k - (k - 1) ** 2 / (n - 1)
    else:
        phi2corr, rcorr, kcorr = 0.0, r, k

    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    valid = ~(categories.isna() | values.isna())
    categories = categories[valid]
    values = values[valid]
    if len(values) == 0:
        return np.nan

    groups = values.groupby(categories)
    if groups.ngroups <= 1:
        return 0.0

    overall_mean = values.mean()
    ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for _, g in groups)
    ss_total = ((values - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return float(a.corr(b, method="spearman"))


def build_association_matrix(df: pd.DataFrame, numeric_cols, cat_cols) -> pd.DataFrame:
    cols = list(numeric_cols) + list(cat_cols)
    M = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    np.fill_diagonal(M.values, 1.0)

    # numeric-numeric
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            v = _spearman_corr(df[c1], df[c2])
            M.loc[c1, c2] = v
            M.loc[c2, c1] = v

    # cat-cat
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i + 1:]:
            v = _cramers_v(df[c1].astype("category"), df[c2].astype("category"))
            M.loc[c1, c2] = v
            M.loc[c2, c1] = v

    # cat-numeric
    for c_cat in cat_cols:
        for c_num in numeric_cols:
            v = _correlation_ratio(df[c_cat].astype("category"),
                                   pd.to_numeric(df[c_num], errors="coerce"))
            M.loc[c_cat, c_num] = v
            M.loc[c_num, c_cat] = v

    return M


def plot_association_heatmap(df_s: pd.DataFrame, out_path: Path) -> None:
    exclude = {"ID"}
    force_categorical = {"DayofWeek", "Period"}

    df = df_s.copy()
    cols = [c for c in df.columns if c not in exclude]

    cat_cols = [c for c in cols if c in force_categorical]
    num_cols = [c for c in cols if c not in force_categorical and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols += [c for c in cols if c not in force_categorical and not pd.api.types.is_numeric_dtype(df[c])]

    for c in cat_cols:
        df[c] = df[c].astype("category")

    M = build_association_matrix(df, numeric_cols=num_cols, cat_cols=cat_cols)

    vals = M.values.astype(float)
    n = vals.shape[0]
    vals_abs = np.abs(vals)
    vmax = float(np.nanquantile(vals_abs[np.isfinite(vals_abs)], 0.95))
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)

    base = plt.get_cmap("YlGnBu")
    cmap = ListedColormap(base(np.linspace(0.15, 0.95, 256)))

    fig, ax = plt.subplots(figsize=(max(9.5, n * 0.75), max(7.5, n * 0.7)))
    im = ax.imshow(vals_abs, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(M.columns, rotation=30, ha="right")
    ax.set_yticklabels(M.index)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("AED sample (n=400): variable association heatmap",
                 fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Association strength", labelpad=10)
    cbar.set_ticks(np.linspace(0, vmax, 5))

    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            txt_color = "white" if vals_abs[i, j] > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=8, fontweight="semibold",
                    color=txt_color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Task 4.2 entry ----------

def run_task4_2(df_s: pd.DataFrame, out_dir: Path, seed: int = 20251222) -> None:
    """
    Task 4.2: interesting relationships between variables
    Inputs:
      - df_s: your unified sampled dataframe (already standardized)
      - out_dir: OUT_TASK4 (outputs/task4)
      - seed: only used for jitter, not sampling
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_day_trend_crowding(
        df_s,
        out_path=out_dir / "task4_2_day_trend_crowding.png",
        roll_win=7
    )

    plot_interesting_relationships_los(
        df_s,
        out_path=out_dir / "task4_2_interesting_relationships_los.png",
        seed=seed
    )

    plot_workload_breach_by_dayofweek(
        df_s,
        out_path=out_dir / "task4_2_workload_breach_by_dayofweek.png"
    )

    plot_arrival_period_relationships(
        df_s,
        out_path=out_dir / "task4_2_arrival_period_relationships.png"
    )

    plot_breach_by_period(
        df_s,
        out_path=out_dir / "task4_2_breach_by_period.png"
    )

    plot_association_heatmap(
        df_s,
        out_path=out_dir / "task4_2_variable_association_heatmap.png"
    )
