# app.py
from __future__ import annotations
import pandas as pd
import streamlit as st
import re
import io
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task1_3 import (
    solve_schedule_task1_3,
    hours_table_to_long_schedule,
    add_cost_column,
    get_total_cost_from_long,
    get_operator_list_long,
    get_operator_hours_long,
    get_day_list_long,
    get_day_schedule_long,
    get_operator_schedule_long,
    WAGES,
)
from src.task4 import run_task4_1, run_task4_2, task4_summary_table, prepare_base

from src.task5 import (
    breach_rate_overall,
    breach_rate_by_category,
    numeric_summary_by_breach,
    top_categories_by_breach_rate,
    get_numeric_vars,
    get_cat_vars,
)

from src.task6 import (
    train_task6_models,
    evaluate_bundle,
    fig_roc_models,
    fig_confusion_matrix_for_model,
    save_task6_outputs,
)

from src.task8_data import (
    LOG_PATH,
    log_event as audit_log_event,
    retrieve_by_id,
    filter_range,
    delete_by_patient_id,
    modify_value_by_patient_id,
)

OUT_BASE = Path("outputs")

OUT_TASK4 = OUT_BASE / "task4"
OUT_TASK5 = OUT_BASE / "task5"
OUT_TASK6 = Path("outputs") / "task6"

OUT_TASK4.mkdir(parents=True, exist_ok=True)
OUT_TASK5.mkdir(parents=True, exist_ok=True)
OUT_TASK6.mkdir(parents=True, exist_ok=True)

# -----------------------
# Config (you can tweak)
# -----------------------
APP_TITLE = "BMAN73701 Coursework UI"
DEFAULT_DATA_PATH = Path("data/AED4weeks_full.csv")

DEFAULT_SEED = 20251222
DEFAULT_SAMPLE_N = 400

THRESH_MIN = 0.05
THRESH_MAX = 0.07
THRESH_DEFAULT = 0.06



FIGS_41 = {
    "Age composite": "task4_age_composite_figure.png",
    "Top HRG (Top 10)": "task4_hrg_top10.png",
    "Crowding overview": "task4_crowding_overview.png",
    "LoS vs Investigation": "task4_los_vs_investigation.png",
    "Breach vs Crowding": "task4_breach_vs_crowding.png",
}

FIGS_42 = {
    "Day trend crowding": "task4_2_day_trend_crowding.png",
    "Interesting relationships (LoS)": "task4_2_interesting_relationships_los.png",
    "Workload & breach by day of week": "task4_2_workload_breach_by_dayofweek.png",
    "Arrival period relationships": "task4_2_arrival_period_relationships.png",
    "Breach by period": "task4_2_breach_by_period.png",
    "Association heatmap": "task4_2_variable_association_heatmap.png",
}


# -----------------------
# Helpers
# -----------------------
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(int(n), len(df))
    return df.sample(n=n, random_state=int(seed)).reset_index(drop=True)


def init_state():
    if "df_full" not in st.session_state:
        if DEFAULT_DATA_PATH.exists():
            st.session_state.df_full = load_csv(DEFAULT_DATA_PATH)
        else:
            st.session_state.df_full = pd.DataFrame()

    if "df_work" not in st.session_state:
        st.session_state.df_work = pd.DataFrame()

    if "sample_n" not in st.session_state:
        st.session_state.sample_n = DEFAULT_SAMPLE_N

    if "sample_seed" not in st.session_state:
        st.session_state.sample_seed = DEFAULT_SEED

    if "threshold" not in st.session_state:
        st.session_state.threshold = THRESH_DEFAULT

    if "df_sample" not in st.session_state:
        if len(st.session_state.df_full) > 0:
            st.session_state.df_sample = sample_df(
                st.session_state.df_full,
                st.session_state.sample_n,
                st.session_state.sample_seed,
            )
        else:
            st.session_state.df_sample = pd.DataFrame()
    if "schedule_df" not in st.session_state:
        st.session_state.schedule_df = pd.DataFrame()
    if "schedule_result" not in st.session_state:
        st.session_state.schedule_result = None  # ScheduleResult or None

    if "schedule_long" not in st.session_state:
        st.session_state.schedule_long = pd.DataFrame()

    if "schedule_results" not in st.session_state:
        st.session_state.schedule_results = {}  # dict: key -> ScheduleResult

    if "schedule_longs" not in st.session_state:
        st.session_state.schedule_longs = {}  # dict: key -> long_df (operator/day/hours/cost)
    if "df_work" not in st.session_state:
        st.session_state.df_work = st.session_state.df_full.copy() if len(st.session_state.df_full) else pd.DataFrame()


def download_file_button(label: str, path: Path, mime: str):
    """Render a download button for a file if it exists."""
    if not path.exists():
        st.caption(f"File not found: {path.name}")
        return
    data = path.read_bytes()
    st.download_button(
        label=label,
        data=data,
        file_name=path.name,
        mime=mime,
    )
def download_file_button(label: str, path: Path, mime: str, key: str):
    if not path.exists():
        st.info("File not generated yet.")
        return
    st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime, key=key)


def show_selected_images_with_download(fig_map: dict, out_dir: Path, key_prefix: str):
    labels = list(fig_map.keys())
    picked = st.multiselect(
        "Select figures to display",
        options=labels,
        default=[],
        key=f"{key_prefix}_pick",
    )

    if not picked:
        st.info("Select one or more figures above.")
        return

    for lab in picked:
        p = out_dir / fig_map[lab]
        st.markdown(f"**{lab}**")

        if p.exists():
            st.image(str(p), use_container_width=True)

            # download button for this PNG
            download_file_button(
                label=f"Download PNG — {p.name}",
                path=p,
                mime="image/png",
            )
        else:
            st.warning(f"Not generated yet: {p.name}. Click the 'Run' button first.")

def download_df_button(label: str, df: pd.DataFrame, file_name: str, key: str):
    if df is None or df.empty:
        st.info("Nothing to download.")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=file_name, mime="text/csv", key=key)

def make_breach_rate_bar_fig(tbl: pd.DataFrame, cat_col: str, top_k: int = 15):
    t = tbl.sort_values("breach_rate", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(t[cat_col].astype(str), t["breach_rate"] * 100)
    ax.set_ylabel("Breach rate (%)")
    ax.set_xlabel(cat_col)
    ax.set_title(f"Breach rate by {cat_col} (top {top_k})")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def make_numeric_boxplot_fig(df_s: pd.DataFrame, num_col: str):
    d = prepare_base(df_s)
    d[num_col] = pd.to_numeric(d[num_col], errors="coerce")
    d = d.dropna(subset=["Breach01", num_col])

    breach = d.loc[d["Breach01"] == 1, num_col].values
    non = d.loc[d["Breach01"] == 0, num_col].values

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot([breach, non], labels=["Breach", "No breach"], showfliers=True)
    ax.set_ylabel(num_col)
    ax.set_title(f"{num_col} by breach status")
    plt.tight_layout()
    return fig


def render_fig_and_optionally_save(fig, save_png: bool, out_path: Path):
    st.pyplot(fig)
    if save_png:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def download_png_button(label: str, png_path: Path, key: str):
    if not png_path.exists():
        st.info("PNG not generated yet.")
        return
    data = png_path.read_bytes()
    st.download_button(label, data=data, file_name=png_path.name, mime="image/png", key=key)

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))


def read_jsonl_log(path: Path) -> pd.DataFrame:
    """Read JSONL log file and flatten details.* into columns."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)  # details.xxx -> details.xxx columns
    return df

def make_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")








@st.cache_data(show_spinner=True)
def solve_schedule_cached(which: str):
    return solve_schedule_task1_3(which)

def clear_schedule_cache():
    solve_schedule_cached.clear()

@st.cache_data(show_spinner=True)
def solve_one_cached(which: str):
    return solve_schedule_task1_3(which)

def render_schedule_result(title: str, res):
    st.subheader(title)
    st.write(res.meta)
    st.metric("Total cost", f"£{res.total_cost:,.2f}")
    st.dataframe(res.hours_table)




def do_resample():
    if len(st.session_state.df_full) == 0:
        st.warning("No full dataset loaded yet.")
        return
    st.session_state.df_sample = sample_df(
        st.session_state.df_full,
        st.session_state.sample_n,
        st.session_state.sample_seed,
    )
    st.success("Resampled successfully ✅")


# -----------------------
# UI: Sidebar
# -----------------------
def render_sidebar() -> str:
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Home", "Task1", "Task2", "Task3", "Task4", "Task5", "Task6", "Task8 (Data)"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Data source")

    # Upload option (optional but nice)
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.df_full = pd.read_csv(uploaded)
            st.session_state.df_sample = sample_df(
                st.session_state.df_full,
                st.session_state.sample_n,
                st.session_state.sample_seed,
            )

            # df work有数值的时候
            st.session_state.df_work = st.session_state.df_full.copy()

            st.sidebar.success("Uploaded & sampled ✅ (df_work initialised)")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")

    st.sidebar.caption(f"Default path: {DEFAULT_DATA_PATH}")

    st.sidebar.divider()
    st.sidebar.subheader("Global controls")

    st.session_state.sample_n = st.sidebar.number_input(
        "Sample size (n)",
        min_value=50,
        max_value=5000,
        value=int(st.session_state.sample_n),
        step=50,
    )

    st.session_state.sample_seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=10**9,
        value=int(st.session_state.sample_seed),
        step=1,
    )

    # threshold for Task6
    st.session_state.threshold = st.sidebar.slider(
        "Task6 threshold",
        min_value=float(THRESH_MIN),
        max_value=float(THRESH_MAX),
        value=float(st.session_state.threshold),
        step=0.001,
    )
    st.sidebar.caption("Recommended: 0.05–0.07 (default 0.06).")

    if st.sidebar.button("Resample"):
        do_resample()

    # -----------------------------
    # Scheduling (Task1–3)
    # -----------------------------
    #st.sidebar.divider()
    #st.sidebar.subheader("Scheduling (Task1–3)")

    #which = st.sidebar.selectbox(
    #    "Choose result to solve/show",
    #   ["task1", "task2_i", "task2_ii", "task3"],
    #    format_func=lambda x: {
    #        "task1": "Task1 (Min Cost)",
    #        "task2_i": "Task2 Scenario i (Fairness + Cost Cap)",
    #        "task2_ii": "Task2 Scenario ii (Min Cost given Best Spread)",
    #        "task3": "Task3 (Skills constraints)",
    #    }[x],
    #)

    #colA, colB = st.sidebar.columns(2)
    #if colA.button("Solve LP"):
    #    try:
    #        res = solve_schedule_cached(which)

    #        # 1) 存 ScheduleResult
    #        st.session_state.schedule_results[which] = res

    #         2) 生成 long schedule 并存
     #       long_df = hours_table_to_long_schedule(res.hours_table)
     #       st.session_state.schedule_longs[which] = add_cost_column(long_df, WAGES)  # 或 wages

    #        st.sidebar.success("Solved ✅")
    #    except Exception as e:
    #        st.sidebar.error(f"Solve failed: {e}")

    #if colB.button("Clear cache"):
     #   solve_schedule_cached.clear()
    #    st.sidebar.info("Cache cleared.")

    # dataset quick stats
    full_rows = len(st.session_state.df_full)
    sample_rows = len(st.session_state.df_sample)
    st.sidebar.info(f"Full rows: {full_rows}\n\nSample rows: {sample_rows}")


    return page


# -----------------------
# Pages
# -----------------------




def page_home():
    st.title(APP_TITLE)
    st.write("Use the left navigation to open Task pages. Sidebar controls apply globally.")

    st.subheader("Current configuration")
    st.write(
        {
            "sample_n": st.session_state.sample_n,
            "sample_seed": st.session_state.sample_seed,
            "task6_threshold": st.session_state.threshold,
        }
    )

    st.subheader("Data preview (full)")
    if len(st.session_state.df_full) == 0:
        st.warning("No dataset loaded. Put CSV at data/AED4weeks_full.csv or upload from sidebar.")
    else:
        st.dataframe(st.session_state.df_full.head(9000))

    st.subheader("Data preview (current sample)")
    if len(st.session_state.df_sample) > 0:
        st.dataframe(st.session_state.df_sample.head(500))

def page_task1():
    st.header("Task 1 — Cost minimisation (baseline)")

    colA, colB = st.columns([1, 1])

    # -------------------------
    # Solve
    # -------------------------
    if colA.button("Solve Task1 (baseline)", key="solve_t1"):
        try:
            res = solve_one_cached("task1")

            # ✅ 统一存到 schedule_results
            st.session_state.schedule_results["task1"] = res

            # ✅ 只在 Solve 时生成 long，并统一存到 schedule_longs
            long_df = hours_table_to_long_schedule(res.hours_table)
            st.session_state.schedule_longs["task1"] = add_cost_column(long_df, WAGES)

            st.success("Solved Task1 ✅")
        except Exception as e:
            st.error(f"Solve failed: {e}")

    if colB.button("Clear cache (Task1)", key="clear_t1"):
        solve_one_cached.clear()
        # 可选：把结果也清掉，避免显示旧结果
        st.session_state.schedule_results.pop("task1", None)
        st.session_state.schedule_longs.pop("task1", None)
        st.info("Cache cleared.")

    # -------------------------
    # Read result
    # -------------------------
    res = st.session_state.schedule_results.get("task1")
    if res is None:
        st.warning("Click 'Solve Task1 (baseline)' to compute the schedule.")
        return

    render_schedule_result("Baseline schedule", res)

    # -------------------------
    # Interactive long view
    # -------------------------
    st.divider()
    st.subheader("Interactive view (filter by operator/day)")

    schedule_long = st.session_state.schedule_longs.get("task1", pd.DataFrame())
    if schedule_long.empty:
        st.info("No long-schedule available yet. Click Solve first.")
        return

    ops = get_operator_list_long(schedule_long)
    days = get_day_list_long(schedule_long)

    c1, c2 = st.columns(2)
    op = c1.selectbox("Operator", ops, key="t1_op")
    day = c2.selectbox("Day", days, key="t1_day")

    if op:
        st.metric("Operator total hours", f"{get_operator_hours_long(schedule_long, op):.2f}")
        st.dataframe(get_operator_schedule_long(schedule_long, op))

    if day:
        st.dataframe(get_day_schedule_long(schedule_long, day))

def page_task2():
    st.header("Task 2 — Fairness constraints (Scenario i & ii)")

    # ---------- Buttons ----------
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])

    def _store_result(which: str, res):
        """Store res + long schedule into session_state under consistent keys."""
        st.session_state.schedule_results[which] = res

        long_df = hours_table_to_long_schedule(res.hours_table)
        st.session_state.schedule_longs[which] = add_cost_column(long_df, WAGES)

    if colA.button("Solve baseline (Task1)", key="solve_t2_base"):
        try:
            res = solve_one_cached("task1")
            _store_result("task1", res)
            st.success("Baseline solved ✅")
        except Exception as e:
            st.error(f"Baseline solve failed: {e}")

    if colB.button("Solve Scenario i", key="solve_t2_i"):
        try:
            res = solve_one_cached("task2_i")
            _store_result("task2_i", res)
            st.success("Task2 scenario i solved ✅")
        except Exception as e:
            st.error(f"Scenario i solve failed: {e}")

    if colC.button("Solve Scenario ii", key="solve_t2_ii"):
        try:
            res = solve_one_cached("task2_ii")
            _store_result("task2_ii", res)
            st.success("Task2 scenario ii solved ✅")
        except Exception as e:
            st.error(f"Scenario ii solve failed: {e}")

    if colD.button("Clear cache (Task2)", key="clear_t2"):
        solve_one_cached.clear()
        # optional: clear cached results shown on this page
        for k in ["task1", "task2_i", "task2_ii"]:
            st.session_state.schedule_results.pop(k, None)
            st.session_state.schedule_longs.pop(k, None)
        st.info("Cache cleared (and Task2 results cleared).")

    st.caption("Tip: solve baseline first for clearer cost comparison.")

    # ---------- Read results ----------
    base = st.session_state.schedule_results.get("task1")
    r_i = st.session_state.schedule_results.get("task2_i")
    r_ii = st.session_state.schedule_results.get("task2_ii")

    if base is None and r_i is None and r_ii is None:
        st.warning("Solve at least one result above.")
        return

    # ---------- Summary ----------
    st.subheader("Comparison summary")
    cols = st.columns(3)

    if base is not None:
        cols[0].metric("Baseline cost", f"£{base.total_cost:,.2f}")
    else:
        cols[0].info("Baseline not solved yet.")

    if r_i is not None:
        cols[1].metric("Scenario i cost", f"£{r_i.total_cost:,.2f}")
        spread_i = r_i.meta.get("fairness_spread")
        if spread_i is not None:
            cols[1].caption(f"Spread (Hmax−Hmin): {spread_i:.2f}")

        # show cost change vs baseline (prefer meta, otherwise compute)
        if "cost_increase_pct" in r_i.meta:
            cols[1].caption(f"Cost change vs baseline: +{r_i.meta['cost_increase_pct']:.2f}%")
        elif base is not None:
            pct = (r_i.total_cost / base.total_cost - 1) * 100
            cols[1].caption(f"Cost change vs baseline: {pct:+.2f}%")
    else:
        cols[1].info("Scenario i not solved yet.")

    if r_ii is not None:
        cols[2].metric("Scenario ii cost", f"£{r_ii.total_cost:,.2f}")
        spread_ii = r_ii.meta.get("fairness_spread")
        if spread_ii is not None:
            cols[2].caption(f"Spread (Hmax−Hmin): {spread_ii:.2f}")

        if base is not None:
            pct = (r_ii.total_cost / base.total_cost - 1) * 100
            cols[2].caption(f"Cost change vs baseline: {pct:+.2f}%")
    else:
        cols[2].info("Scenario ii not solved yet.")

    # ---------- Tables ----------
    st.divider()
    if base is not None:
        render_schedule_result("Baseline schedule (Task1)", base)

    if r_i is not None:
        render_schedule_result("Task2 Scenario i (fairness + cost cap)", r_i)

    if r_ii is not None:
        render_schedule_result("Task2 Scenario ii (min cost given best spread)", r_ii)

    # ---------- Operator totals side-by-side ----------
    st.divider()
    st.subheader("Operator weekly totals (side-by-side)")

    def totals(res, label: str):
        df = res.hours_table.copy()
        if "Total" not in df.columns:
            # assume schedule table is numeric columns Mon..Fri
            df["Total"] = df.sum(axis=1)
        return df[["Total"]].rename(columns={"Total": label})

    frames = []
    if base is not None:
        frames.append(totals(base, "Baseline"))
    if r_i is not None:
        frames.append(totals(r_i, "Scenario i"))
    if r_ii is not None:
        frames.append(totals(r_ii, "Scenario ii"))

    if frames:
        comp = pd.concat(frames, axis=1)
        st.dataframe(comp)




def page_task3():
    st.header("Task 3 — Skills constraints")

    colA, colB = st.columns([1, 1])

    def _store_result(which: str, res):
        st.session_state.schedule_results[which] = res
        long_df = hours_table_to_long_schedule(res.hours_table)
        st.session_state.schedule_longs[which] = add_cost_column(long_df, WAGES)

    if colA.button("Solve Task3 (skills)", key="solve_t3"):
        try:
            res = solve_one_cached("task3")   # ✅ 你的 task1_3.py 就是这个 key
            _store_result("task3", res)
            st.success("Task3 solved ✅")
        except Exception as e:
            st.error(f"Task3 solve failed: {e}")

    if colB.button("Clear cache (Task3)", key="clear_t3"):
        solve_one_cached.clear()
        st.session_state.schedule_results.pop("task3", None)
        st.session_state.schedule_longs.pop("task3", None)
        st.info("Cache cleared (Task3 cleared).")

    res = st.session_state.schedule_results.get("task3")
    if res is None:
        st.warning("Click 'Solve Task3 (skills)' to compute the schedule.")
        return

    # 结果表
    render_schedule_result("Task3 schedule", res)

    # 和 baseline 对比（可选，但很建议）
    base = st.session_state.schedule_results.get("task1")
    if base is not None:
        st.divider()
        st.subheader("Cost comparison vs baseline")
        pct = (res.total_cost / base.total_cost - 1) * 100
        c1, c2 = st.columns(2)
        c1.metric("Baseline cost", f"£{base.total_cost:,.2f}")
        c2.metric("Task3 cost", f"£{res.total_cost:,.2f}", delta=f"{pct:+.2f}%")
    else:
        st.info("Tip: Solve Task1 baseline first if you want cost comparison.")









def _img(p: Path):
    """Safely display image if exists."""
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.info(f"Not generated yet: {p.name}")

def show_selected_images(fig_map: dict, out_dir: Path, key_prefix: str):
    labels = list(fig_map.keys())

    picked = st.multiselect(
        "Select figures to display",
        options=labels,
        default=[],
        key=f"{key_prefix}_pick",
    )

    if not picked:
        st.info("Select one or more figures above.")
        return

    for lab in picked:
        p = out_dir / fig_map[lab]
        st.markdown(f"**{lab}**")
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.warning(f"Not generated yet: {p.name}. Click the 'Run' button first.")


def page_task4():
    st.header("Task 4 — Answering questions (AED exploration)")

    df = st.session_state.df_sample
    if df is None or df.empty:
        st.warning("No sample data available. Please load data / resample in the sidebar.")
        return

    st.caption(f"Current sample size: n = {len(df)}")
    with st.expander("Preview sample (df_sample)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    # -----------------------------
    # Question-driven UI
    # -----------------------------
    questions = {
        "Q1 — Generate all Task 4.1 outputs (summary table + main figures)": "t4_q1",
        "Q2 — What are the main numeric characteristics of this sample? (Summary table)": "t4_q2",
        "Q3 — Show main-feature plots (choose which figures to display)": "t4_q3",
        "Q4 — Generate all Task 4.2 relationship outputs (figures)": "t4_q4",
        "Q5 — Explore relationships (choose which figures to display)": "t4_q5",
        "Q6 — Download Task 4 outputs (Excel/CSV/PNGs)": "t4_q6",
    }

    q = st.selectbox("Choose a question", list(questions.keys()), key="t4_choose_q")

    st.divider()
    st.markdown(f"### **Question**")
    st.write(q)

    st.markdown("### **Answer**")

    # ---------- Shared paths ----------
    xlsx_path = OUT_TASK4 / "Task4_summary_table.xlsx"
    csv_path = OUT_TASK4 / "Task4_summary_table.csv"

    # ============= Q1: run Task4.1 =============
    if q.startswith("Q1"):
        st.info("This will generate the Task 4.1 summary table and figures into outputs/task4/.")

        if st.button("Run Task 4.1 (generate outputs)", key="run_t4_1"):
            try:
                run_task4_1(df, OUT_TASK4)
                st.success("Task 4.1 outputs generated ✅")
            except Exception as e:
                st.error(f"Task 4.1 failed: {e}")

        st.caption("After running, you can go to Q2/Q3/Q6 to view/download results.")

    # ============= Q2: summary table in-app =============
    elif q.startswith("Q2"):
        st.caption("This computes the summary table in-app (and also exports Excel/CSV to outputs/task4/).")

        if st.button("Compute & show summary table", key="show_t4_summary"):
            try:
                dprep = prepare_base(df)
                summary = task4_summary_table(
                    dprep,
                    out_dir=OUT_TASK4,
                    out_name="Task4_summary_table.xlsx",
                    save_csv=True,
                )
                st.dataframe(summary, use_container_width=True)
                st.success("Summary table generated ✅")
                st.caption("Exported to outputs/task4/Task4_summary_table.xlsx (+ CSV).")
            except Exception as e:
                st.error(f"Summary table failed: {e}")

        # quick download if exists
        st.divider()
        st.subheader("Downloads (if already generated)")
        c1, c2 = st.columns(2)
        with c1:
            if xlsx_path.exists():
                st.download_button(
                    "⬇️ Download summary (Excel)",
                    data=xlsx_path.read_bytes(),
                    file_name=xlsx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="t4_dl_xlsx_q2",
                )
            else:
                st.caption("Excel not found yet. Run Q1 or click 'Compute & show summary table'.")
        with c2:
            if csv_path.exists():
                st.download_button(
                    "⬇️ Download summary (CSV)",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    key="t4_dl_csv_q2",
                )
            else:
                st.caption("CSV not found yet. Run Q1 or click 'Compute & show summary table'.")

    # ============= Q3: show figures 4.1 =============
    elif q.startswith("Q3"):
        st.caption("Choose which Task 4.1 figures you want to display (must exist in outputs/task4/).")
        show_selected_images(FIGS_41, OUT_TASK4, key_prefix="t41_qa")

        st.divider()
        st.caption("Tip: if images are missing, run Q1 first to generate outputs.")

    # ============= Q4: run Task4.2 =============
    elif q.startswith("Q4"):
        seed = int(st.session_state.get("sample_seed", 20251222))
        st.info(f"This will generate Task 4.2 relationship figures into outputs/task4/. (jitter seed={seed})")

        if st.button("Run Task 4.2 (generate outputs)", key="run_t4_2"):
            try:
                run_task4_2(df, OUT_TASK4, seed=seed)
                st.success("Task 4.2 outputs generated ✅")
            except Exception as e:
                st.error(f"Task 4.2 failed: {e}")

        st.caption("After running, go to Q5/Q6 to view/download.")

    # ============= Q5: show figures 4.2 =============
    elif q.startswith("Q5"):
        st.caption("Choose which Task 4.2 relationship figures you want to display (must exist in outputs/task4/).")
        show_selected_images(FIGS_42, OUT_TASK4, key_prefix="t42_qa")

        st.divider()
        st.caption("Tip: if images are missing, run Q4 first to generate outputs.")

    # ============= Q6: downloads =============
    elif q.startswith("Q6"):
        st.subheader("Summary table downloads")
        c1, c2 = st.columns(2)
        with c1:
            if xlsx_path.exists():
                st.download_button(
                    "⬇️ Download summary (Excel)",
                    data=xlsx_path.read_bytes(),
                    file_name=xlsx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="t4_dl_xlsx_q6",
                )
            else:
                st.caption("Excel not generated yet (run Q1 or Q2).")

        with c2:
            if csv_path.exists():
                st.download_button(
                    "⬇️ Download summary (CSV)",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    key="t4_dl_csv_q6",
                )
            else:
                st.caption("CSV not generated yet (run Q1 or Q2).")

        st.divider()
        st.subheader("Figure downloads (PNG)")

        # Build a simple list of known figure files from your FIGS_41/FIGS_42 configs
        # Assumption: FIGS_41 / FIGS_42 is a list/dict that contains filenames used by show_selected_images.
        # If your FIGS_* structure is dict {label: filename}, this works.
        fig_files = []
        if isinstance(FIGS_41, dict):
            fig_files += list(FIGS_41.values())
        if isinstance(FIGS_42, dict):
            fig_files += list(FIGS_42.values())
        fig_files = [str(x) for x in fig_files]

        # fallback: show a manual download for everything that exists in OUT_TASK4/*.png
        existing_pngs = sorted([p for p in OUT_TASK4.glob("*.png")])

        if existing_pngs:
            pick = st.selectbox(
                "Choose a PNG to download",
                options=[p.name for p in existing_pngs],
                key="t4_png_pick",
            )
            p = OUT_TASK4 / pick
            st.download_button(
                f"⬇️ Download {pick}",
                data=p.read_bytes(),
                file_name=p.name,
                mime="image/png",
                key="t4_dl_png_one",
            )
        else:
            st.caption("No PNG files found in outputs/task4 yet. Run Q1/Q4 first.")



def page_task5():
    st.header("Task 5 — Answering questions: What drives breaches?")

    df = st.session_state.df_sample
    if df is None or df.empty:
        st.warning("No sample data available. Please load data / resample in the sidebar.")
        return

    st.caption(f"Current sample size: n = {len(df)}")
    with st.expander("Preview sample", expanded=False):
        st.dataframe(df.head(30), use_container_width=True)

    st.divider()
    st.subheader("Choose a question")

    questions = [
        "Q1 — What is the overall breach rate in the current sample?",
        "Q2 — Which categories have the highest breach rate? (categorical factors)",
        "Q3 — Do breached vs non-breached cases differ on a numeric variable?",
        "Q4 — Generate a quick output pack (tables/plots) for reporting",
    ]
    q = st.selectbox("Choose a question", questions, key="t5_q_select")

    st.divider()

    # -------------------------
    # Q1 Overall breach rate
    # -------------------------
    if q.startswith("Q1"):
        st.markdown("### Question")
        st.write("**What is the overall breach rate in the current sample?**")

        st.markdown("### Answer")
        m = breach_rate_overall(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Valid n", f"{m['n_valid']}")
        c2.metric("Breach count", f"{m['breach']}")
        c3.metric("Breach rate", f"{m['breach_rate']*100:.2f}%")

        st.caption("Tip: This uses the sample currently loaded in session_state.df_sample.")

    # -------------------------
    # Q2 Categorical factors
    # -------------------------
    elif q.startswith("Q2"):
        st.markdown("### Question")
        st.write("**Which categories have the highest breach rate?**")

        st.markdown("### Choose inputs")
        cat_vars = get_cat_vars(df)
        if not cat_vars:
            st.info("No categorical variables found among defaults (DayofWeek/Period/HRG).")
            return

        cat = st.selectbox("Choose a categorical variable", cat_vars, key="t5_cat")
        top_k = st.slider("Top K categories to show", 5, 30, 15, 1, key="t5_cat_topk")

        st.markdown("### Answer")
        tbl = breach_rate_by_category(df, cat)

        # table
        st.write("**Breach rate table**")
        st.dataframe(tbl, use_container_width=True)

        # chart + optional save
        st.write("**Visualisation (bar chart)**")
        save_png = st.checkbox("Save bar chart to outputs/task5 (PNG)", value=False, key="t5_cat_savepng")
        fig = make_breach_rate_bar_fig(tbl, cat_col=cat, top_k=top_k)

        png_path = OUT_TASK5 / f"task5_breach_rate_by_{cat}_top{top_k}.png"
        render_fig_and_optionally_save(fig, save_png=save_png, out_path=png_path)

        # downloads
        st.divider()
        st.subheader("Downloads")
        download_df_button(
            "Download categorical table (CSV)",
            tbl,
            file_name=f"task5_breach_by_{cat}.csv",
            key="t5_cat_dl",
        )
        download_png_button("Download bar chart (PNG)", png_path, key="t5_cat_png_dl")

    # -------------------------
    # Q3 Numeric factors
    # -------------------------
    elif q.startswith("Q3"):
        st.markdown("### Question")
        st.write("**Do breached vs non-breached cases differ on a numeric variable?**")

        st.markdown("### Choose inputs")
        num_vars = get_numeric_vars(df)
        if not num_vars:
            st.info("No numeric variables found among defaults (Age/LoS/noofpatients/...).")
            return

        num = st.selectbox("Choose a numeric variable", num_vars, key="t5_num")

        st.markdown("### Answer")
        tbl2 = numeric_summary_by_breach(df, num)

        st.write("**Summary table (Breach vs No breach)**")
        st.dataframe(tbl2, use_container_width=True)

        st.write("**Visualisation (boxplot)**")
        save_png = st.checkbox("Save boxplot to outputs/task5 (PNG)", value=False, key="t5_num_savepng")
        fig = make_numeric_boxplot_fig(df, num_col=num)

        png_path = OUT_TASK5 / f"task5_boxplot_{num}.png"
        render_fig_and_optionally_save(fig, save_png=save_png, out_path=png_path)

        # downloads
        st.divider()
        st.subheader("Downloads")
        download_df_button(
            "Download numeric summary (CSV)",
            tbl2,
            file_name=f"task5_numeric_summary_{num}.csv",
            key="t5_num_dl",
        )
        download_png_button("Download boxplot (PNG)", png_path, key="t5_num_png_dl")

    # -------------------------
    # Q4 Output pack
    # -------------------------
    else:
        st.markdown("### Question")
        st.write("**Can I generate a quick set of outputs (tables) for reporting?**")

        st.markdown("### Choose inputs")
        cat_vars = get_cat_vars(df)
        num_vars = get_numeric_vars(df)

        c1, c2 = st.columns(2)
        with c1:
            pack_cat = st.selectbox("Categorical variable (optional)", cat_vars, key="t5_pack_cat") if cat_vars else None
            top_k = st.slider("Top K (categorical)", 5, 20, 10, 1, key="t5_pack_topk")
        with c2:
            pack_num = st.selectbox("Numeric variable (optional)", num_vars, key="t5_pack_num") if num_vars else None

        st.markdown("### Answer")
        if st.button("Generate pack", key="t5_pack_btn"):
            if not pack_cat and not pack_num:
                st.warning("Please choose at least one variable (categorical or numeric).")
                return

            if pack_cat:
                st.markdown("#### Top categories by breach rate")
                top_tbl = top_categories_by_breach_rate(df, pack_cat, top_k=top_k)
                st.dataframe(top_tbl, use_container_width=True)
                download_df_button(
                    "Download top categories (CSV)",
                    top_tbl,
                    file_name=f"task5_top_categories_{pack_cat}.csv",
                    key="t5_pack_cat_dl",
                )

            if pack_num:
                st.markdown("#### Numeric summary by breach")
                num_tbl = numeric_summary_by_breach(df, pack_num)
                st.dataframe(num_tbl, use_container_width=True)
                download_df_button(
                    "Download numeric pack summary (CSV)",
                    num_tbl,
                    file_name=f"task5_numeric_pack_{pack_num}.csv",
                    key="t5_pack_num_dl",
                )


def page_task6():
    st.header("Task 6 — Predict breach (threshold tuning)")

    # dataset choose: Full vs Sample
    eval_source = st.radio("Dataset for training/evaluation", ["Full dataset", "Sample (sidebar)"], horizontal=True)
    df = st.session_state.df_full if eval_source == "Full dataset" else st.session_state.df_sample

    if df is None or df.empty:
        st.warning("Selected dataset is empty. Please load data first.")
        return

    st.caption(f"Using: **{eval_source}** | n = {len(df)}")

    # training settings
    c1, c2, c3 = st.columns(3)
    seed = c1.number_input("Seed", min_value=0, max_value=10**9, value=int(st.session_state.get("sample_seed", 42)), step=1)
    test_size = c2.slider("Test size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    use_xgb = c3.checkbox("Use XGB (if installed)", value=True)

    thr = float(st.session_state.get("threshold", 0.06))
    st.caption(f"Threshold from sidebar: **{thr:.3f}** (recommended 0.05–0.07)")

    # Train button (avoid retrain on every rerun)
    train_btn = st.button("Train models (LogReg / RF / XGB)", key="t6_train")

    # cache in session_state
    cache_key = f"{eval_source}|seed={seed}|test={test_size}|xgb={use_xgb}"
    if "task6_bundle" not in st.session_state:
        st.session_state.task6_bundle = {}
    bundles = st.session_state.task6_bundle

    if train_btn:
        try:
            bundle = train_task6_models(df, seed=int(seed), test_size=float(test_size), use_xgb=bool(use_xgb))
            bundles[cache_key] = bundle
            st.success("Training complete ✅ (stored in session)")
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

    bundle = bundles.get(cache_key, None)
    if bundle is None:
        st.info("Click **Train models** first (training uses the selected dataset).")
        return

    # Evaluate at current threshold (no retrain)
    metrics_df = evaluate_bundle(bundle, threshold=thr)

    st.subheader("Metrics (at current threshold)")
    st.dataframe(metrics_df)

    st.divider()
    st.subheader("ROC (all models)")
    fig_roc = fig_roc_models(bundle, metrics_df, threshold=thr)
    st.pyplot(fig_roc)

    st.divider()
    st.subheader("Confusion matrix (choose model)")
    model_names = list(metrics_df["model"].astype(str))
    chosen = st.selectbox("Model", model_names, index=0)
    fig_cm = fig_confusion_matrix_for_model(metrics_df, chosen)
    st.pyplot(fig_cm)

    # Save + downloads
    st.divider()
    st.subheader("Save & Download")

    tag = "full" if eval_source == "Full dataset" else "sample"

    save_all = st.checkbox("Save outputs to outputs/task6/", value=True, key="t6_save_all")
    saved_paths = {}

    if save_all:
        try:
            # also save CM for all models (not only chosen) so downloads are complete
            figs_cm_all = {m: fig_confusion_matrix_for_model(metrics_df, m) for m in model_names}
            saved_paths = save_task6_outputs(
                OUT_TASK6,
                metrics_df=metrics_df,
                fig_roc=fig_roc,
                figs_cm=figs_cm_all,
                tag=tag,
                threshold=thr,
            )
            st.success("Saved ✅")
        except Exception as e:
            st.warning(f"Save failed: {e}")

    # Downloads (only appear if saved)
    if saved_paths:
        cA, cB = st.columns(2)
        with cA:
            download_file_button("Download metrics (CSV)", saved_paths["metrics_csv"], "text/csv", key="t6_dl_metrics")
        with cB:
            download_file_button("Download ROC (PNG)", saved_paths["roc_png"], "image/png", key="t6_dl_roc")

        st.caption("Confusion matrices (PNG):")
        for m in model_names:
            k = f"cm_png_{m}"
            if k in saved_paths:
                download_file_button(f"Download CM — {m}", saved_paths[k], "image/png", key=f"t6_dl_cm_{m}")
    else:
        st.info("Tick 'Save outputs' to enable downloads.")


OUT_TASK8 = Path("outputs") / "task8"
log_event = OUT_TASK8 / "audit_log.jsonl"


def page_task8_data():
    st.header("Task 8 — Data Management (CRUD) + Logging")

    # 一个全局 safe_mode 就够了（Delete/Modify 都用它）
    safe_mode = st.checkbox(
        "I understand this will permanently change df_work (Delete/Modify).",
        value=False,
        key="t8_safe_global"
    )

    if st.session_state.df_work is None or st.session_state.df_work.empty:
        st.warning("df_work is empty. Load data first, then you can manage it here.")
        return

    dfw = st.session_state.df_work
    st.caption(f"Working dataset rows: {len(dfw)}")

    with st.expander("Preview (df_work)", expanded=False):
        st.dataframe(dfw.head(30), use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Retrieve", "Range filter", "Delete", "Modify", "Logs"])

    # -------- Retrieve --------
    with tab1:
        st.subheader("Retrieve by patient ID")

        cols = list(dfw.columns)
        default_idx = cols.index("ID") if "ID" in cols else 0
        id_col = st.selectbox("ID column", options=cols, index=default_idx, key="t8_id_col")

        pid = st.text_input("Patient ID", key="t8_pid")

        if st.button("Search", key="t8_search"):
            try:
                out = retrieve_by_id(dfw, id_col=id_col, value=pid)
                st.success(f"Found {len(out)} row(s).")
                st.dataframe(out, use_container_width=True)

                audit_log_event(
                    action="QUERY_BY_ID",
                    success=True,
                    details={"id_col": id_col, "value": pid, "rows": len(out)},
                )
            except Exception as e:
                st.error(str(e))
                audit_log_event(
                    action="QUERY_BY_ID",
                    success=False,
                    details={"id_col": id_col, "value": pid},
                    error=str(e),
                )

    # -------- Range filter --------
    with tab2:
        st.subheader("Filter by numeric range")

        numeric_candidates = [c for c in
                              ["Age", "LoS", "noofpatients", "noofinvestigation", "nooftreatment", "Day", "Period"]
                              if c in dfw.columns]
        if not numeric_candidates:
            st.warning("No numeric columns found.")
        else:
            col = st.selectbox("Numeric column", options=numeric_candidates, key="t8_range_col")

            c1, c2 = st.columns(2)
            vmin = c1.number_input("Min", value=0.0, key="t8_min")
            vmax = c2.number_input("Max", value=100.0, key="t8_max")

            if st.button("Apply filter", key="t8_apply_filter"):
                try:
                    out = filter_range(dfw, col=col, vmin=float(vmin), vmax=float(vmax))
                    st.success(f"Matched {len(out)} row(s).")
                    st.dataframe(out.head(300), use_container_width=True)

                    st.download_button(
                        "Download filtered CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name=f"task8_filter_{col}_{vmin}_{vmax}.csv",
                        mime="text/csv",
                        key="t8_dl_filter"
                    )

                    audit_log_event(
                        action="FILTER_RANGE",
                        success=True,
                        details={"col": col, "min": float(vmin), "max": float(vmax), "rows": len(out)},
                    )
                except Exception as e:
                    st.error(str(e))
                    audit_log_event(
                        action="FILTER_RANGE",
                        success=False,
                        details={"col": col, "min": float(vmin), "max": float(vmax)},
                        error=str(e),
                    )

    # -------- Delete --------
    with tab3:
        st.subheader("Delete records by patient ID")

        cols = list(dfw.columns)
        default_idx = cols.index("ID") if "ID" in cols else 0
        id_col = st.selectbox("ID column", options=cols, index=default_idx, key="t8_del_id_col")
        pid_del = st.text_input("Patient ID to delete", key="t8_del_pid")

        if st.button("Delete", key="t8_delete"):
            if not safe_mode:
                st.warning("Please tick the safety checkbox first.")
            else:
                try:
                    before = len(dfw)
                    new_df, deleted = delete_by_patient_id(dfw, patient_id=pid_del, id_col=id_col)
                    st.session_state.df_work = new_df
                    after = len(new_df)

                    st.success(f"Deleted {deleted} row(s). Rows: {before} → {after}")

                    audit_log_event(
                        action="DELETE",
                        success=True,
                        details={
                            "id_col": id_col,
                            "patient_id": pid_del,
                            "deleted": deleted,
                            "rows_before": before,
                            "rows_after": after,
                        },
                    )
                except Exception as e:
                    st.error(str(e))
                    audit_log_event(
                        action="DELETE",
                        success=False,
                        details={"id_col": id_col, "patient_id": pid_del},
                        error=str(e),
                    )

    # -------- Modify --------
    with tab4:
        st.subheader("Modify / Update by patient ID")

        cols = list(dfw.columns)
        default_idx = cols.index("ID") if "ID" in cols else 0
        id_col = st.selectbox("ID column", options=cols, index=default_idx, key="t8_mod_idcol")

        pid_mod = st.text_input("Patient ID to modify", key="t8_mod_pid")

        editable_cols = [c for c in dfw.columns if c != id_col]
        col = st.selectbox("Column to modify", options=editable_cols, key="t8_mod_col")
        new_val = st.text_input("New value", key="t8_mod_val")

        if st.button("Update", key="t8_update"):
            if not safe_mode:
                st.warning("Please tick the safety checkbox first.")
            else:
                try:
                    new_df, old_vals = modify_value_by_patient_id(
                        dfw, patient_id=pid_mod, id_col=id_col, col=col, new_value=new_val
                    )
                    st.session_state.df_work = new_df
                    st.success(f"Updated ✅ (matched rows: {len(old_vals)})")

                    audit_log_event(
                        action="MODIFY",
                        success=True,
                        details={
                            "id_col": id_col,
                            "patient_id": pid_mod,
                            "col": col,
                            "affected": len(old_vals),
                            "old_sample": old_vals[:10],  # 日志别太大
                            "new": new_val,
                        },
                    )
                except Exception as e:
                    st.error(str(e))
                    audit_log_event(
                        action="MODIFY",
                        success=False,
                        details={"id_col": id_col, "patient_id": pid_mod, "col": col, "new": new_val},
                        error=str(e),
                    )

    # -------- Logs --------
    with tab5:
        st.subheader("Audit log (JSONL → table)")

        log_df = read_jsonl_log(LOG_PATH)

        if log_df.empty:
            st.info(f"No log file yet: {LOG_PATH}")
            st.caption("Tip: run a Search / Filter / Delete / Modify to generate log entries.")
            return

        st.caption(f"Log file: {LOG_PATH} | rows: {len(log_df)}")

        # Filters
        c1, c2, c3 = st.columns(3)
        with c1:
            action_opts = ["(All)"] + sorted(log_df["action"].dropna().astype(str).unique().tolist())
            action_sel = st.selectbox("Action", action_opts, key="t8_log_action")
        with c2:
            succ_sel = st.selectbox("Success", ["(All)", "True", "False"], key="t8_log_success")
        with c3:
            keyword = st.text_input("Keyword (search in details/error)", key="t8_log_kw")

        view = log_df.copy()
        if action_sel != "(All)":
            view = view[view["action"].astype(str) == action_sel]

        if succ_sel != "(All)":
            want = True if succ_sel == "True" else False
            view = view[view["success"].astype(bool) == want]

        if keyword.strip():
            kw = keyword.strip().lower()
            cols = [c for c in view.columns if c == "error" or c.startswith("details.")]
            if cols:
                mask = False
                for c in cols:
                    mask = mask | view[c].astype(str).str.lower().str.contains(kw, na=False)
                view = view[mask]

        # sort by time desc if present
        if "ts_utc" in view.columns:
            view = view.sort_values("ts_utc", ascending=False)

        st.dataframe(view, use_container_width=True)

        st.divider()
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button(
                "Download filtered logs (CSV)",
                data=make_download_bytes(view),
                file_name="task8_logs_filtered.csv",
                mime="text/csv",
                key="t8_dl_logs_filtered",
            )
        with cdl2:
            st.download_button(
                "Download ALL logs (CSV)",
                data=make_download_bytes(log_df),
                file_name="task8_logs_all.csv",
                mime="text/csv",
                key="t8_dl_logs_all",
            )

        if st.button("Refresh log view", key="t8_log_refresh"):
            st.rerun()







def placeholder_page(name: str):
    st.header(f"{name} — placeholder")
    st.write("We will connect your existing task outputs here.")
    st.caption("For now, the global sample + threshold controls already work.")


# -----------------------
# Main
# -----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    page = render_sidebar()

    if page == "Home":
        page_home()


    elif page == "Task1":

        page_task1()

    elif page == "Task2":

        page_task2()

    elif page == "Task3":

        page_task3()

    elif page == "Task4":
        page_task4()

    elif page == "Task5":
        page_task5()

    elif page == "Task6":
        page_task6()

    elif page == "Task8 (Data)":
        page_task8_data()



    else:
        placeholder_page(page)


if __name__ == "__main__":
    main()
