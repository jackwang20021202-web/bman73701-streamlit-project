# src/task1_3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import pandas as pd
import pulp

WAGES = {
    "E.Khan": 25,
    "Y.Chen": 26,
    "A.Taylor": 24,
    "R.Zidane": 23,
    "R.Perez": 28,
    "C.Santos": 30,
}

@dataclass
class ScheduleResult:
    key: str
    total_cost: float
    hours_table: pd.DataFrame   # wide table: index=operator, columns=Mon..Fri + Total
    meta: dict


class Scheduler:
    """
    Task 1: Total cost minimisation
    Task 2: Fairness constraints (scenario i & ii)
    Task 3: Skills constraints
    """

    def __init__(self) -> None:
        self.days: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        self.operators: List[str] = ["E.Khan", "Y.Chen", "A.Taylor", "R.Zidane", "R.Perez", "C.Santos"]

        self.availability: Dict[str, Dict[str, float]] = {
            "E.Khan": {"Mon": 6, "Tue": 0, "Wed": 6, "Thu": 0, "Fri": 6},
            "Y.Chen": {"Mon": 0, "Tue": 6, "Wed": 0, "Thu": 6, "Fri": 0},
            "A.Taylor": {"Mon": 4, "Tue": 8, "Wed": 4, "Thu": 0, "Fri": 4},
            "R.Zidane": {"Mon": 5, "Tue": 5, "Wed": 5, "Thu": 0, "Fri": 5},
            "R.Perez": {"Mon": 3, "Tue": 0, "Wed": 3, "Thu": 8, "Fri": 0},
            "C.Santos": {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 6, "Fri": 2},
        }

        self.wages = WAGES

        self.min_weekly_hours: Dict[str, float] = {
            "E.Khan": 8.0,
            "Y.Chen": 8.0,
            "A.Taylor": 8.0,
            "R.Zidane": 8.0,
            "R.Perez": 7.0,
            "C.Santos": 7.0,
        }

        self.daily_demand: float = 14.0

        self.skills: List[str] = ["Programming", "Troubleshooting"]
        self.skills_map: Dict[str, Set[str]] = {
            "E.Khan": {"Programming"},
            "Y.Chen": {"Programming"},
            "A.Taylor": {"Troubleshooting"},
            "R.Zidane": {"Troubleshooting"},
            "R.Perez": {"Programming"},
            "C.Santos": {"Programming", "Troubleshooting"},
        }

        self.results: Dict[str, ScheduleResult] = {}

    def _add_total_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self.days].copy()
        df = df.round(1)
        df["Total"] = df.sum(axis=1).round(1)
        return df

    def _solve_or_raise(self, model: pulp.LpProblem, solver: Optional[pulp.LpSolver] = None) -> str:
        solver = solver or pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        status = pulp.LpStatus[model.status]
        if status not in {"Optimal", "Feasible"}:
            raise RuntimeError(f"Optimization failed. Status={status}")
        return status

    def build_baseline_model(self, name: str):
        model = pulp.LpProblem(name, pulp.LpMinimize)
        h = pulp.LpVariable.dicts("h", (self.operators, self.days), lowBound=0, cat="Continuous")

        # C1 availability: h[i,d] <= availability[i][d]
        for i in self.operators:
            for d in self.days:
                model += h[i][d] <= self.availability[i][d]

        # C2 min weekly hours
        for i in self.operators:
            model += pulp.lpSum(h[i][d] for d in self.days) >= self.min_weekly_hours[i]

        # C3 daily coverage
        for d in self.days:
            model += pulp.lpSum(h[i][d] for i in self.operators) == self.daily_demand

        return model, h

    def solve_task1(self) -> ScheduleResult:
        model, h = self.build_baseline_model("Task1_CostMin")
        total_cost_expr = pulp.lpSum(self.wages[i] * h[i][d] for i in self.operators for d in self.days)
        model += total_cost_expr

        status = self._solve_or_raise(model)

        total_cost = float(pulp.value(total_cost_expr))
        df = pd.DataFrame({d: [pulp.value(h[i][d]) for i in self.operators] for d in self.days}, index=self.operators)
        df = self._add_total_column(df)

        res = ScheduleResult(key="task1", total_cost=total_cost, hours_table=df, meta={"status": status})
        self.results[res.key] = res
        return res

    def solve_task2_scenario_1(self, baseline_cost: float) -> ScheduleResult:
        model, h = self.build_baseline_model("Task2_S1_Fairness_CostCap")

        H = {i: pulp.lpSum(h[i][d] for d in self.days) for i in self.operators}
        H_max = pulp.LpVariable("H_max", lowBound=0)
        H_min = pulp.LpVariable("H_min", lowBound=0)
        for i in self.operators:
            model += H[i] <= H_max
            model += H[i] >= H_min

        total_cost_expr = pulp.lpSum(self.wages[i] * h[i][d] for i in self.operators for d in self.days)
        model += total_cost_expr <= 1.018 * baseline_cost

        # objective: minimise spread
        model += (H_max - H_min)

        status = self._solve_or_raise(model)

        spread = float(pulp.value(H_max - H_min))
        total_cost = float(pulp.value(total_cost_expr))
        df = pd.DataFrame({d: [pulp.value(h[i][d]) for i in self.operators] for d in self.days}, index=self.operators)
        df = self._add_total_column(df)

        res = ScheduleResult(
            key="task2_i",
            total_cost=total_cost,
            hours_table=df,
            meta={
                "status": status,
                "fairness_spread": spread,
                "baseline_cost": baseline_cost,
                "cost_increase_pct": (total_cost / baseline_cost - 1) * 100,
            },
        )
        self.results[res.key] = res
        return res

    def solve_task2_scenario_2(self, baseline_cost: float) -> ScheduleResult:
        # Stage 1: find best spread
        model1, h1 = self.build_baseline_model("Task2_S2_Stage1_MinSpread")
        H1 = {i: pulp.lpSum(h1[i][d] for d in self.days) for i in self.operators}
        H_max1 = pulp.LpVariable("H_max1", lowBound=0)
        H_min1 = pulp.LpVariable("H_min1", lowBound=0)
        for i in self.operators:
            model1 += H1[i] <= H_max1
            model1 += H1[i] >= H_min1
        model1 += (H_max1 - H_min1)

        _ = self._solve_or_raise(model1)
        best_spread = float(pulp.value(H_max1 - H_min1))

        # Stage 2: minimise cost given spread
        model2, h2 = self.build_baseline_model("Task2_S2_Stage2_MinCost_GivenSpread")
        H2 = {i: pulp.lpSum(h2[i][d] for d in self.days) for i in self.operators}
        H_max2 = pulp.LpVariable("H_max2", lowBound=0)
        H_min2 = pulp.LpVariable("H_min2", lowBound=0)
        for i in self.operators:
            model2 += H2[i] <= H_max2
            model2 += H2[i] >= H_min2
        model2 += (H_max2 - H_min2) <= best_spread

        total_cost_expr = pulp.lpSum(self.wages[i] * h2[i][d] for i in self.operators for d in self.days)
        model2 += total_cost_expr

        status = self._solve_or_raise(model2)

        total_cost = float(pulp.value(total_cost_expr))
        df = pd.DataFrame({d: [pulp.value(h2[i][d]) for i in self.operators] for d in self.days}, index=self.operators)
        df = self._add_total_column(df)

        res = ScheduleResult(
            key="task2_ii",
            total_cost=total_cost,
            hours_table=df,
            meta={"status": status, "fairness_spread": best_spread, "baseline_cost": baseline_cost},
        )
        self.results[res.key] = res
        return res

    def solve_task3(self, baseline_cost: float) -> ScheduleResult:
        model, h = self.build_baseline_model("Task3_Skills")

        total_cost_expr = pulp.lpSum(self.wages[i] * h[i][d] for i in self.operators for d in self.days)
        model += total_cost_expr

        # skill coverage: at least 6 hours per skill per day
        skill_ops = {s: [i for i in self.operators if s in self.skills_map[i]] for s in self.skills}
        for d in self.days:
            for s in self.skills:
                model += pulp.lpSum(h[i][d] for i in skill_ops[s]) >= 6

        status = self._solve_or_raise(model)

        total_cost = float(pulp.value(total_cost_expr))
        df = pd.DataFrame({d: [pulp.value(h[i][d]) for i in self.operators] for d in self.days}, index=self.operators)
        df = self._add_total_column(df)

        res = ScheduleResult(
            key="task3",
            total_cost=total_cost,
            hours_table=df,
            meta={"status": status, "baseline_cost": baseline_cost, "cost_increase_pct": (total_cost / baseline_cost - 1) * 100},
        )
        self.results[res.key] = res
        return res

    def run_all(self) -> Dict[str, ScheduleResult]:
        r1 = self.solve_task1()
        self.solve_task2_scenario_1(r1.total_cost)
        self.solve_task2_scenario_2(r1.total_cost)
        self.solve_task3(r1.total_cost)
        return self.results


# -------------------------
# UI-friendly wrapper APIs
# -------------------------

def solve_schedule_task1_3(which: str) -> ScheduleResult:
    """
    which in {"task1", "task2_i", "task2_ii", "task3"}.
    We compute baseline first when needed.
    """
    sch = Scheduler()

    if which == "task1":
        return sch.solve_task1()

    r1 = sch.solve_task1()
    baseline_cost = r1.total_cost

    if which == "task2_i":
        return sch.solve_task2_scenario_1(baseline_cost)
    if which == "task2_ii":
        return sch.solve_task2_scenario_2(baseline_cost)
    if which == "task3":
        return sch.solve_task3(baseline_cost)

    raise ValueError(f"Unknown option: {which}")


def hours_table_to_long_schedule(hours_table: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide hours table (index=operator, cols=Mon..Fri, Total)
    to a long table with columns: operator, day, hours.
    """
    df = hours_table.copy()
    df.index.name = "operator"
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])
    long_df = df.reset_index().melt(id_vars=["operator"], var_name="day", value_name="hours")
    long_df["hours"] = long_df["hours"].astype(float)
    return long_df


def add_cost_column(schedule_long: pd.DataFrame, wages: Dict[str, float]) -> pd.DataFrame:
    """
    Add cost=hours*wage to long schedule.
    """
    df = schedule_long.copy()
    df["wage"] = df["operator"].map(wages).astype(float)
    df["cost"] = df["hours"] * df["wage"]
    return df


# Query helpers for UI (work on long schedule)
def get_total_cost_from_long(schedule_long: pd.DataFrame) -> float:
    if schedule_long is None or schedule_long.empty:
        return 0.0
    if "cost" not in schedule_long.columns:
        raise KeyError("schedule_long missing 'cost' column")
    return float(schedule_long["cost"].sum())


def get_operator_list_long(schedule_long: pd.DataFrame) -> list[str]:
    if schedule_long is None or schedule_long.empty:
        return []
    return sorted(schedule_long["operator"].dropna().astype(str).unique().tolist())


def get_day_list_long(schedule_long: pd.DataFrame) -> list[str]:
    if schedule_long is None or schedule_long.empty:
        return []
    days = schedule_long["day"].dropna().astype(str).unique().tolist()
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return sorted(days, key=lambda x: order.index(x) if x in order else 999)


def get_operator_hours_long(schedule_long: pd.DataFrame, operator: str) -> float:
    if schedule_long is None or schedule_long.empty:
        return 0.0
    sub = schedule_long[schedule_long["operator"].astype(str) == str(operator)]
    return float(sub["hours"].sum())


def get_day_schedule_long(schedule_long: pd.DataFrame, day: str) -> pd.DataFrame:
    if schedule_long is None or schedule_long.empty:
        return pd.DataFrame()
    return schedule_long[schedule_long["day"].astype(str) == str(day)].copy()


def get_operator_schedule_long(schedule_long: pd.DataFrame, operator: str) -> pd.DataFrame:
    if schedule_long is None or schedule_long.empty:
        return pd.DataFrame()
    return schedule_long[schedule_long["operator"].astype(str) == str(operator)].copy()
