# src/task5_los.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# -------------------------
# Shared helpers
# -------------------------
def _find_col(df: pd.DataFrame, keywords: List[str]) -> str:
    """Find a column by fuzzy keyword match (case/space/punct insensitive)."""
    norm = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in df.columns}
    for k in keywords:
        kk = re.sub(r"[^a-z0-9]+", "", k.lower())
        for c, nc in norm.items():
            if kk == nc or kk in nc:
                return c
    raise KeyError(f"Could not find columns for keywords={keywords}. Available={list(df.columns)}")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def wilson_ci_2(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for proportion. Returns (lo, hi)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stratified_sample_min_each(
    df: pd.DataFrame,
    group_col: str,
    n_total: int,
    min_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified sampling with minimum per group."""
    rng = np.random.default_rng(seed)

    base_parts = []
    sizes = df[group_col].value_counts(dropna=False)

    for g, sub in df.groupby(group_col, sort=False):
        take = min(len(sub), min_per_group)
        if take > 0:
            base_parts.append(sub.sample(n=take, random_state=int(rng.integers(1, 1_000_000_000))))
    base = pd.concat(base_parts, ignore_index=True) if base_parts else df.iloc[0:0].copy()

    if len(base) >= n_total:
        return base.sample(n=n_total, random_state=int(rng.integers(1, 1_000_000_000)))

    remaining = n_total - len(base)
    weights = sizes / sizes.sum()
    alloc = (weights * remaining).round().astype(int)

    diff = remaining - int(alloc.sum())
    if diff != 0:
        order = weights.sort_values(ascending=False).index.tolist()
        step = 1 if diff > 0 else -1
        i = 0
        while diff != 0 and i < 10000:
            g = order[i % len(order)]
            alloc[g] = max(0, int(alloc.get(g, 0)) + step)
            diff -= step
            i += 1

    extra_parts = []
    for g, sub in df.groupby(group_col, sort=False):
        take = int(alloc.get(g, 0))
        if take <= 0:
            continue
        take = min(take, len(sub))
        extra_parts.append(sub.sample(n=take, random_state=int(rng.integers(1, 1_000_000_000))))
    extra = pd.concat(extra_parts, ignore_index=True) if extra_parts else df.iloc[0:0].copy()

    out = pd.concat([base, extra], ignore_index=True)
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=int(rng.integers(1, 1_000_000_000)))
    return out


# -------------------------
# Plot 1: HRG vs LoS + prolonged risk
# -------------------------
def plot_hrg_los_iqr_and_prolonged_risk(
    df_s: pd.DataFrame,
    out_dir: Path,
    breach_minutes: int = 240,
    n_sample: int = 400,
    seed: int = 20251229,
    min_per_hrg: int = 10,
    min_n_to_plot: int = 1,
    out_name: str = "task5_HRG_LoS_IQR_and_prolonged_risk.png",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hrg_col = _find_col(df_s, ["HRG"])
    los_col = _find_col(df_s, ["LoS", "LengthofStay", "length_of_stay"])

    d_all = df_s[[hrg_col, los_col]].copy()
    d_all[los_col] = _to_num(d_all[los_col])
    d_all = d_all.dropna(subset=[hrg_col, los_col])
    d_all = d_all[d_all[los_col] >= 0].copy()

    d = stratified_sample_min_each(
        d_all, group_col=hrg_col,
        n_total=min(n_sample, len(d_all)),
        min_per_group=min_per_hrg,
        seed=seed
    )

    cnt = d[hrg_col].value_counts()
    keep = cnt[cnt >= min_n_to_plot].index
    d = d[d[hrg_col].isin(keep)].copy()

    rows = []
    for h, g in d.groupby(hrg_col, observed=True):
        x = g[los_col].to_numpy()
        n = len(x)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        k = int(np.sum(x > breach_minutes))
        rate = k / n if n else np.nan
        lo, hi = wilson_ci_2(k, n)
        rows.append([h, n, q1, med, q3, k, rate, lo, hi])

    S = pd.DataFrame(rows, columns=["HRG", "n", "q1", "median", "q3", "k_breach", "rate", "ci_lo", "ci_hi"])
    S = S.sort_values("n", ascending=False).reset_index(drop=True)

    cmap = LinearSegmentedColormap.from_list(
        "blue_teal_yellowgreen",
        ["#0B2A6F", "#0A6FB6", "#07F5F1", "#7CFFB2", "#D6F01A"],
        N=256,
    )
    norm = Normalize(vmin=float(S["median"].min()), vmax=float(S["median"].max()))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = [cmap(norm(m)) for m in S["median"].to_numpy()]

    fig = plt.figure(figsize=(13.6, 6.9), dpi=220)
    fig.subplots_adjust(top=0.84)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.95, 0.05], wspace=0.12)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    BAR_LW = 6.2
    DOT_S = 62

    for i in range(len(S)):
        axA.hlines(i, S.loc[i, "q1"], S.loc[i, "q3"], lw=BAR_LW, color=colors[i], alpha=0.95)
        axA.scatter(S.loc[i, "median"], i, s=DOT_S, color=colors[i],
                    edgecolor="white", linewidth=1.0, zorder=3)

        axB.hlines(i, S.loc[i, "ci_lo"], S.loc[i, "ci_hi"], lw=BAR_LW, color=colors[i], alpha=0.95)
        axB.scatter(S.loc[i, "rate"], i, s=DOT_S, color=colors[i],
                    edgecolor="white", linewidth=1.0, zorder=3)

    y = np.arange(len(S))
    labels = [f"{h}  (n={n})" for h, n in zip(S["HRG"], S["n"])]
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=10.5)
    axB.set_yticks(y)
    axB.set_yticklabels([""] * len(y))

    for ax in (axA, axB):
        ax.grid(axis="x", color="#AAB4BF", alpha=0.25, linestyle="--", linewidth=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "HRG and Length of Stay: distributional differences + prolonged-stay signal",
        fontsize=17, fontweight="bold", y=0.965
    )
    axA.set_title("A. LoS by HRG: median (dot) and IQR (bar)", fontsize=13, fontweight="bold", pad=10)
    axB.set_title(f"B. Prolonged-stay risk (LoS > {breach_minutes} min): rate ± 95% CI",
                  fontsize=13, fontweight="bold", pad=10)

    axA.set_xlabel("Length of Stay (minutes)", fontsize=11.5)
    axB.set_xlabel("Proportion", fontsize=11.5)

    axA.set_xlim(left=0)
    axB.set_xlim(0, min(1.0, max(0.55, float(S["ci_hi"].max()) * 1.08)))

    axA.invert_yaxis()
    axB.invert_yaxis()

    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Median LoS (minutes)", fontsize=11.5)
    cb.ax.tick_params(labelsize=10.5)

    fig.text(0.985, 0.935, f"Total n={len(d)} (sampled)", ha="right", va="top", fontsize=10.5, alpha=0.75)

    out_path = out_dir / out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------
# Plot 2: Investigation vs LoS (0–8) + prolonged risk
# -------------------------
def plot_investigation_los_0to8(
    df_s: pd.DataFrame,
    out_dir: Path,
    breach_minutes: int = 240,
    n_sample: int = 400,
    seed: int = 20251229,
    inv_levels: Optional[List[int]] = None,
    min_per_level: int = 12,
    out_name_A: str = "task5_investigation_LoS_distribution_0to8.png",
    out_name_B: str = "task5_investigation_prolonged_risk_0to8.png",
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if inv_levels is None:
        inv_levels = list(range(0, 9))

    los_col = _find_col(df_s, ["LoS", "LengthofStay", "length_of_stay"])
    inv_col = _find_col(df_s, ["noofinvestigation", "investigation", "numinvestigation"])

    d_all = df_s[[los_col, inv_col]].copy()
    d_all[los_col] = _to_num(d_all[los_col])
    d_all[inv_col] = _to_num(d_all[inv_col])
    d_all = d_all.dropna(subset=[los_col, inv_col])
    d_all = d_all[(d_all[los_col] >= 0) & (d_all[inv_col] >= 0)].copy()

    d_all["inv_int"] = np.floor(d_all[inv_col]).astype(int)
    d_all = d_all[d_all["inv_int"].isin(inv_levels)].copy()

    d = stratified_sample_min_each(
        d_all, group_col="inv_int",
        n_total=min(n_sample, len(d_all)),
        min_per_group=min_per_level,
        seed=seed
    )

    rows = []
    for lv in inv_levels:
        g = d[d["inv_int"] == lv]
        x = g[los_col].to_numpy()
        n = len(x)
        if n == 0:
            rows.append([lv, 0, np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
            continue
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        k = int(np.sum(x > breach_minutes))
        rate = k / n
        lo, hi = wilson_ci_2(k, n)
        rows.append([lv, n, q1, med, q3, k, rate, lo, hi])

    S = pd.DataFrame(rows, columns=["inv", "n", "q1", "median", "q3", "k_breach", "rate", "ci_lo", "ci_hi"])

    cmap = LinearSegmentedColormap.from_list(
        "pink_purple_science",
        ["#2B0A57", "#5B2A86", "#9B5DE5", "#F15BB5", "#F7A8C9"],
        N=256
    )
    valid = S["median"].dropna()
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = [cmap(norm(m)) if np.isfinite(m) else (0.85, 0.85, 0.85, 1.0) for m in S["median"].to_numpy()]

    # ---- Fig A: box + jitter
    figA = plt.figure(figsize=(13.2, 6.8), dpi=220)
    ax = figA.add_subplot(111)
    figA.subplots_adjust(top=0.86)

    pos = np.arange(len(inv_levels))
    data_by_lv = [d.loc[d["inv_int"] == lv, los_col].to_numpy() for lv in inv_levels]

    bp = ax.boxplot(
        data_by_lv, positions=pos, widths=0.58, patch_artist=True, showfliers=False,
        medianprops=dict(color="white", linewidth=2.0),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.72)
        patch.set_edgecolor("#0B0F14")

    rng = np.random.default_rng(seed)
    for i, lv in enumerate(inv_levels):
        yvals = d.loc[d["inv_int"] == lv, los_col].to_numpy()
        if len(yvals) == 0:
            continue
        xjit = rng.normal(loc=i, scale=0.065, size=len(yvals))
        ax.scatter(xjit, yvals, s=18, alpha=0.18, edgecolor="none", color=colors[i])

    ax.axhline(breach_minutes, linestyle="--", linewidth=1.7, alpha=0.65)
    ax.text(len(inv_levels)-0.05, breach_minutes, f"  breach = {breach_minutes} min",
            va="center", ha="left", fontsize=10.5, alpha=0.75)

    xt = []
    for lv in inv_levels:
        n_lv = int(S.loc[S["inv"] == lv, "n"].values[0])
        xt.append(f"{lv}\n(n={n_lv})")
    ax.set_xticks(pos)
    ax.set_xticklabels(xt, fontsize=10.5)

    ax.set_title("LoS distribution by number of investigation (0–8): box + jitter",
                 fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("Number of investigation", fontsize=12)
    ax.set_ylabel("Length of Stay (minutes)", fontsize=12)
    ax.grid(axis="y", color="#AAB4BF", alpha=0.18, linestyle="--", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = figA.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Median LoS (minutes)", fontsize=11.5)
    figA.text(0.985, 0.94, f"Total n={len(d)} (sampled)", ha="right", va="top", fontsize=10.5, alpha=0.75)

    outA = out_dir / out_name_A
    figA.savefig(outA, bbox_inches="tight")
    plt.close(figA)

    # ---- Fig B: risk forest
    figB = plt.figure(figsize=(11.8, 6.6), dpi=220)
    ax2 = figB.add_subplot(111)
    figB.subplots_adjust(top=0.86)

    S2 = S[S["n"] > 0].copy()
    y = np.arange(len(S2))

    for i, (_, row) in enumerate(S2.iterrows()):
        lv = int(row["inv"])
        r = float(row["rate"])
        lo = float(row["ci_lo"])
        hi = float(row["ci_hi"])
        n_lv = int(row["n"])

        idx = inv_levels.index(lv)
        col = colors[idx]

        ax2.hlines(i, lo, hi, lw=6.0, color=col, alpha=0.95)
        ax2.scatter(r, i, s=30 + 9*np.sqrt(n_lv), color=col, edgecolor="white", linewidth=1.0, zorder=3)

    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{int(v)}  (n={int(nv)})" for v, nv in zip(S2["inv"], S2["n"])], fontsize=11)
    ax2.invert_yaxis()

    ax2.set_title(f"Prolonged-stay risk by investigation (0–8): P(LoS > {breach_minutes}) ± 95% CI (Wilson)",
                  fontsize=15, fontweight="bold", pad=10)
    ax2.set_xlabel(f"Proportion with LoS > {breach_minutes} min (±95% CI)", fontsize=12)

    xmax = float(np.nanmax(S2["ci_hi"].to_numpy()))
    ax2.set_xlim(0, min(1.0, max(0.20, xmax * 1.10)))

    ax2.grid(axis="x", color="#AAB4BF", alpha=0.25, linestyle="--", linewidth=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    cbar2 = figB.colorbar(sm, ax=ax2, fraction=0.05, pad=0.04)
    cbar2.set_label("Median LoS (minutes)", fontsize=11.5)
    figB.text(0.985, 0.94, f"Total n={len(d)} (sampled)", ha="right", va="top", fontsize=10.5, alpha=0.75)

    outB = out_dir / out_name_B
    figB.savefig(outB, bbox_inches="tight")
    plt.close(figB)

    return outA, outB


# -------------------------
# Plot 3: Period × crowding quartile — median LoS heatmap + N bars + breach line
# -------------------------
def plot_period_crowdQ_median_los_heatmap(
    df_s: pd.DataFrame,
    out_dir: Path,
    seed: int = 20251222,
    n_sample: int = 400,
    breach_minutes: int = 240,
    min_cell_n: int = 5,
    out_name: str = "task5_LoS_Period_x_CrowdingQuartile_heatmap.png",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    col_period = _find_col(df_s, ["Period"])
    col_los = _find_col(df_s, ["LoS", "LengthofStay", "length_of_stay"])
    col_crowd = _find_col(df_s, ["noofpatients"])

    d = df_s[[col_period, col_los, col_crowd]].copy()
    d[col_period] = _to_num(d[col_period])
    d[col_los] = _to_num(d[col_los])
    d[col_crowd] = _to_num(d[col_crowd])
    d = d.dropna(subset=[col_period, col_los, col_crowd]).copy()

    d = d[(d[col_period] >= 0) & (d[col_period] <= 23)].copy()
    d[col_period] = d[col_period].astype(int)

    if len(d) > n_sample:
        d = d.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    d["breach"] = (d[col_los] > breach_minutes).astype(int)

    labels = ["Low", "Mid-Low", "Mid-High", "High"]
    ranked = d[col_crowd].rank(method="first")
    d["CrowdQ"] = pd.qcut(ranked, q=4, labels=labels)

    g = (
        d.groupby([col_period, "CrowdQ"], observed=True)
        .agg(median_los=(col_los, "median"), n=(col_los, "size"))
        .reset_index()
    )

    periods = list(range(24))
    rows = labels

    mat_med = (
        g.pivot(index="CrowdQ", columns=col_period, values="median_los")
        .reindex(index=rows, columns=periods)
        .astype(float)
    )
    mat_n = (
        g.pivot(index="CrowdQ", columns=col_period, values="n")
        .reindex(index=rows, columns=periods)
        .fillna(0)
        .astype(int)
    )

    mat_show = mat_med.copy()
    mat_show[mat_n < min_cell_n] = np.nan

    per_period = (
        d.groupby(col_period, observed=True)
        .agg(N=(col_los, "size"), breach_rate=("breach", "mean"))
        .reindex(periods)
        .fillna(0)
    )

    v = mat_show.to_numpy().ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(v, [5, 95])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))

    def make_cmap():
        cols = ["#2A0A5E", "#0E5AA7", "#11BFAE", "#7CFF6B", "#FFE45E"]
        cmap = LinearSegmentedColormap.from_list("bright_viridis_like", cols, N=256)
        cmap.set_bad("#FFFFFF")
        return cmap

    def text_color(val: float, vmin_: float, vmax_: float) -> str:
        if not np.isfinite(val):
            return "#6B6B6B"
        t = (val - vmin_) / (vmax_ - vmin_ + 1e-12)
        return "white" if t < 0.45 else "#111111"

    mpl.rcParams.update({"font.size": 11})

    cmap = make_cmap()

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.35])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(
        mat_show.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title("Median LoS by Arrival Period × Crowding Quartile", pad=14)
    ax.set_xlabel("Arrival period (0–23)")
    ax.set_ylabel("Crowding level (quartiles of noofpatients)")

    ax.set_xticks(np.arange(len(periods)))
    ax.set_xticklabels(periods)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)

    ax.set_xticks(np.arange(-.5, len(periods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, qlab in enumerate(rows):
        for j, p in enumerate(periods):
            n = int(mat_n.loc[qlab, p])
            m = float(mat_med.loc[qlab, p]) if np.isfinite(mat_med.loc[qlab, p]) else np.nan

            if (n >= min_cell_n) and np.isfinite(m):
                tc = text_color(m, vmin, vmax)
                ax.text(j, i - 0.05, f"{int(round(m))}", ha="center", va="center",
                        fontsize=10, fontweight="semibold", color=tc)
                ax.text(j, i + 0.28, f"N={n}", ha="center", va="center",
                        fontsize=8, color=tc, alpha=0.95,
                        path_effects=[pe.withStroke(linewidth=2.0,
                                                    foreground="black" if tc == "white" else "white",
                                                    alpha=0.45)])
            elif (0 < n < min_cell_n):
                ax.text(j, i, f"N={n}", ha="center", va="center", fontsize=8, color="#9A9A9A")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Median LoS (minutes)")

    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(periods))
    Ns = per_period["N"].values.astype(float)

    bar_cmap = plt.cm.Blues
    normN = (Ns - Ns.min()) / (Ns.max() - Ns.min() + 1e-12) if Ns.max() > 0 else Ns
    bar_colors = bar_cmap(0.25 + 0.55 * normN)
    ax2.bar(x, Ns, color=bar_colors, edgecolor="#CFE6FF", linewidth=0.6)

    ax2.set_ylabel("Sample size (N)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods)
    ax2.set_xlabel("Arrival period (0–23)")
    ax2.grid(axis="y", color="#E8E8E8", linewidth=0.9)

    ax3 = ax2.twinx()
    ax3.plot(x, per_period["breach_rate"].values, linewidth=2.2, marker="o",
             markersize=4.2, color="#0B2E4E")
    ax3.set_ylabel("Breach rate (proportion)")
    ax3.set_ylim(0, max(0.05, float(per_period["breach_rate"].max()) * 1.25))

    fig.subplots_adjust(left=0.06, right=0.93, top=0.90, bottom=0.09, hspace=0.35)

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------
# Task entry
# -------------------------
def run_task5_los(df_s: pd.DataFrame, out_dir: Path) -> None:
    """Generate the three Task5 LoS figures into out_dir."""
    plot_hrg_los_iqr_and_prolonged_risk(df_s, out_dir)
    plot_investigation_los_0to8(df_s, out_dir)
    plot_period_crowdQ_median_los_heatmap(df_s, out_dir)
