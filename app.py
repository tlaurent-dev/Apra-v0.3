import streamlit as st
import pandas as pd
from datetime import date
import tempfile, os

from apra_core import (
    analyze_project,
    monte_carlo_project_duration,
    summarize_samples,
    make_task_risk_fig,
    make_monte_carlo_fig,
    explain_project_risk,
    recommend_actions,
)

# =========================
# Status + styling
# =========================

def task_status(risk_pct: float, green_lt: float, orange_lt: float) -> str:
    if risk_pct < green_lt:
        return "🟩 On track"
    elif risk_pct < orange_lt:
        return "🟧 Recoverable"
    else:
        return "🟥 High risk"


def project_status(delay_pct: float, green_lt: float, orange_lt: float) -> str:
    if delay_pct < green_lt:
        return "🟩 On track"
    elif delay_pct < orange_lt:
        return "🟧 Recoverable"
    else:
        return "🟥 High risk"


def status_bucket(status_text: str) -> str:
    if "🟩" in status_text:
        return "Green"
    if "🟧" in status_text:
        return "Orange"
    if "🟥" in status_text:
        return "Red"
    return "Unknown"


def style_status_col(df: pd.DataFrame, col_name: str = "Status"):
    def _style(v):
        if isinstance(v, str) and "🟩" in v:
            return "background-color: #DFF2BF;"
        if isinstance(v, str) and "🟧" in v:
            return "background-color: #FEEFB3;"
        if isinstance(v, str) and "🟥" in v:
            return "background-color: #FFBABA;"
        return ""
    return df.style.applymap(_style, subset=[col_name])

# =========================
# Trend history (Step 2)
# =========================

HISTORY_COLUMNS = [
    "Snapshot Date", "Project",
    "Delay %", "P80", "Planned (days)",
    "Max Task Risk %", "Avg Task Risk %",
    "Status", "PERT?"
]

def init_history():
    if "apra_history" not in st.session_state:
        st.session_state["apra_history"] = pd.DataFrame(columns=HISTORY_COLUMNS)

def history_add_snapshot(snapshot_date: date, portfolio_df: pd.DataFrame):
    init_history()
    hist = st.session_state["apra_history"]

    rows = []
    for _, r in portfolio_df.iterrows():
        rows.append({
            "Snapshot Date": snapshot_date.isoformat(),
            "Project": r["Project"],
            "Delay %": float(r["Delay %"]),
            "P80": float(r["P80"]),
            "Planned (days)": float(r["Planned (days)"]),
            "Max Task Risk %": float(r["Max Task Risk %"]),
            "Avg Task Risk %": float(r["Avg Task Risk %"]),
            "Status": r["Status"],
            "PERT?": r.get("PERT?", "No"),
        })

    snap_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)

    if not hist.empty:
        key = ["Snapshot Date", "Project"]
        hist = hist[~hist.set_index(key).index.isin(snap_df.set_index(key).index)]

    hist = pd.concat([hist, snap_df], ignore_index=True)
    hist["Snapshot Date"] = hist["Snapshot Date"].astype(str)
    hist = hist.sort_values(["Snapshot Date", "Project"], ascending=[True, True]).reset_index(drop=True)

    st.session_state["apra_history"] = hist

def history_to_csv_bytes() -> bytes:
    init_history()
    return st.session_state["apra_history"].to_csv(index=False).encode("utf-8")

def history_load_from_uploaded(file) -> None:
    init_history()
    try:
        df = pd.read_csv(file)
        missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Uploaded history CSV is missing columns: {', '.join(missing)}")
            st.stop()
        st.session_state["apra_history"] = df[HISTORY_COLUMNS].copy()
    except Exception as e:
        st.error(f"Failed to load history CSV: {e}")
        st.stop()

def compute_trends(history_df: pd.DataFrame, current_portfolio_df: pd.DataFrame):
    if history_df.empty or current_portfolio_df.empty:
        return pd.DataFrame()

    hist = history_df.copy()
    hist["Snapshot Date"] = pd.to_datetime(hist["Snapshot Date"], errors="coerce")
    hist = hist.dropna(subset=["Snapshot Date"])

    current = current_portfolio_df[["Project", "Delay %", "Status"]].copy()
    current.rename(columns={"Delay %": "Current Delay %", "Status": "Current Status"}, inplace=True)

    rows = []
    for proj in current["Project"].tolist():
        ph = hist[hist["Project"] == proj].sort_values("Snapshot Date")
        if len(ph) < 1:
            continue

        latest = ph.iloc[-1]

        curr_delay = float(current.loc[current["Project"] == proj, "Current Delay %"].iloc[0])
        curr_status = current.loc[current["Project"] == proj, "Current Status"].iloc[0]

        last_delay = float(latest["Delay %"])
        delta = curr_delay - last_delay

        if abs(delta) < 0.75:
            arrow = "→"
        elif delta > 0:
            arrow = "↑"
        else:
            arrow = "↓"

        last_bucket = status_bucket(str(latest["Status"]))
        now_bucket = status_bucket(str(curr_status))
        new_red = (now_bucket == "Red" and last_bucket != "Red")

        rows.append({
            "Project": proj,
            "Current Status": curr_status,
            "Current Delay %": round(curr_delay, 2),
            "Last Snapshot Date": latest["Snapshot Date"].date().isoformat(),
            "Last Delay %": round(last_delay, 2),
            "Δ vs Last": round(delta, 2),
            "Trend": arrow,
            "New Red?": "YES" if new_red else "",
        })

    return pd.DataFrame(rows)

# =========================
# Cost of Delay (Step 4)
# =========================

def init_cost_overrides():
    if "cost_per_day_overrides" not in st.session_state:
        st.session_state["cost_per_day_overrides"] = {}  # {project_name: cost_per_day}

def get_cost_per_day(project_name: str, default_cost: float) -> float:
    init_cost_overrides()
    v = st.session_state["cost_per_day_overrides"].get(project_name, None)
    try:
        if v is None:
            return float(default_cost)
        return float(v)
    except Exception:
        return float(default_cost)

def set_cost_override(project_name: str, cost: float):
    init_cost_overrides()
    st.session_state["cost_per_day_overrides"][project_name] = float(cost)

def expected_delay_days(planned: float, p80: float) -> float:
    return max(0.0, float(p80) - float(planned))

def expected_delay_cost(delay_probability_pct: float, exp_delay_days: float, cost_per_day: float) -> float:
    return (float(delay_probability_pct) / 100.0) * float(exp_delay_days) * float(cost_per_day)

# =========================
# Ownership & Accountability (Step 5)
# =========================

OWNER_COLS = ["Task Owner", "Risk Owner"]

def ensure_owner_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in OWNER_COLS:
        if c not in df2.columns:
            df2[c] = "Unassigned"
    df2["Task Owner"] = df2["Task Owner"].fillna("Unassigned").astype(str).replace({"": "Unassigned"})
    df2["Risk Owner"] = df2["Risk Owner"].fillna("Unassigned").astype(str).replace({"": "Unassigned"})
    return df2

def build_open_items_by_owner(project_name: str, df_tasks: pd.DataFrame,
                              proj_delay_pct: float, proj_expected_cost: float,
                              task_green: float, task_orange: float):
    """
    Returns a task-level table of OPEN items (Orange/Red), with attributed cost.
    Attribution: distribute project expected delay cost across open tasks proportionally by Propagated Risk %.
    """
    d = df_tasks.copy()
    d["Propagated Risk %"] = d["Propagated Risk %"].astype(float)
    d["Progress"] = d["Progress"].astype(float)

    d = ensure_owner_columns(d)

    d["Task Status"] = d["Propagated Risk %"].apply(lambda x: task_status(float(x), task_green, task_orange))
    open_mask = d["Task Status"].isin(["🟧 Recoverable", "🟥 High risk"])
    open_df = d.loc[open_mask].copy()

    if open_df.empty:
        return open_df

    # cost attribution weights
    risk_sum = float(open_df["Propagated Risk %"].sum())
    if risk_sum <= 0:
        open_df["Attributed Cost ($)"] = 0.0
    else:
        open_df["Attributed Cost ($)"] = open_df["Propagated Risk %"].apply(
            lambda r: (float(r) / risk_sum) * float(proj_expected_cost)
        )

    # FIX: handle CSVs that already include a "Project" column
    if "Project" not in open_df.columns:
        open_df.insert(0, "Project", project_name)
    else:
        open_df["Project"] = open_df["Project"].fillna(project_name)

    open_df["Project Delay %"] = float(proj_delay_pct)

    keep = [
        "Project",
        "Project Delay %",
        "Task Status",
        "Task",
        "Task Owner",
        "Risk Owner",
        "Propagated Risk %",
        "Base Risk %",
        "Progress",
        "On Critical Path",
        "Attributed Cost ($)",
        "Start",
        "Due",
    ]
    for c in keep:
        if c not in open_df.columns:
            open_df[c] = ""

    return open_df[keep].sort_values(["Task Status", "Attributed Cost ($)"], ascending=[True, False])

def summarize_open_items_by_owner(open_items_df: pd.DataFrame, owner_field: str):
    if open_items_df.empty:
        return pd.DataFrame()

    d = open_items_df.copy()
    d["Owner"] = d[owner_field].astype(str).fillna("Unassigned").replace({"": "Unassigned"})

    d["Is Red"] = d["Task Status"].apply(lambda s: 1 if "🟥" in str(s) else 0)
    d["Is Orange"] = d["Task Status"].apply(lambda s: 1 if "🟧" in str(s) else 0)

    g = d.groupby("Owner", as_index=False).agg(
        Open_Items=("Task", "count"),
        Red_Items=("Is Red", "sum"),
        Orange_Items=("Is Orange", "sum"),
        Attributed_Cost_USD=("Attributed Cost ($)", "sum"),
        Avg_Task_Risk_Pct=("Propagated Risk %", "mean"),
    )
    g["Attributed_Cost_USD"] = g["Attributed_Cost_USD"].round(2)
    g["Avg_Task_Risk_Pct"] = g["Avg_Task_Risk_Pct"].round(2)

    g = g.sort_values(["Attributed_Cost_USD", "Red_Items"], ascending=[False, False])
    return g

# =========================
# Helpers
# =========================

def analyze_project_from_df(df_in, today):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df_in.to_csv(tmp.name, index=False)
        path = tmp.name
    try:
        return analyze_project(path, today=today)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def validate_csv(df):
    required = ["Task", "Start", "Due", "Progress"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    pert_cols = ["Optimistic", "MostLikely", "Pessimistic"]
    has_any_pert = any(c in df.columns for c in pert_cols)

    if has_any_pert:
        for c in pert_cols:
            if c not in df.columns:
                st.error(f"If you use PERT, your CSV must include all three columns: {', '.join(pert_cols)}")
                st.stop()

        bad_rows = []
        for i, r in df.iterrows():
            vals = [r.get("Optimistic"), r.get("MostLikely"), r.get("Pessimistic")]
            if all(pd.isna(v) or str(v).strip() == "" for v in vals):
                continue
            try:
                o = float(r["Optimistic"])
                m = float(r["MostLikely"])
                p = float(r["Pessimistic"])
                if not (o <= m <= p):
                    bad_rows.append((i + 2, r.get("Task", f"Row{i+2}"),
                                     "Must satisfy Optimistic <= MostLikely <= Pessimistic"))
            except Exception:
                bad_rows.append((i + 2, r.get("Task", f"Row{i+2}"), "PERT values must be numeric"))

        if bad_rows:
            msg = "Invalid PERT rows found (CSV row numbers shown):\n\n"
            msg += "\n".join([f"- Row {rn} ({task}): {reason}" for rn, task, reason in bad_rows[:15]])
            st.error(msg)
            st.stop()


def ensure_pert_columns(df):
    df2 = df.copy()
    for c in ["Optimistic", "MostLikely", "Pessimistic"]:
        if c not in df2.columns:
            df2[c] = ""
    return df2


def init_overrides():
    if "overrides_by_project" not in st.session_state:
        st.session_state["overrides_by_project"] = {}

def get_project_overrides(project_name: str):
    init_overrides()
    return st.session_state["overrides_by_project"].setdefault(project_name, {})

def apply_overrides(df, project_overrides: dict):
    df2 = df.copy()
    if not project_overrides:
        return df2
    for i, r in df2.iterrows():
        t = str(r["Task"])
        if t in project_overrides:
            ov = project_overrides[t]
            for k, v in ov.items():
                df2.at[i, k] = v
    return df2

def any_row_has_pert_values(df):
    cols = ["Optimistic", "MostLikely", "Pessimistic"]
    if not all(c in df.columns for c in cols):
        return False
    for _, r in df.iterrows():
        vals = [r.get(c) for c in cols]
        if all(not (pd.isna(v) or str(v).strip() == "") for v in vals):
            return True
    return False

# =========================
# UI
# =========================

st.set_page_config(page_title="APRA Portfolio Dashboard", layout="wide")
st.title("APRA — Project Risk Dashboard")

init_overrides()
init_history()
init_cost_overrides()

with st.sidebar:
    st.header("Navigation")
    view_mode = st.radio("View", ["Portfolio", "Project details", "Owners"], index=0)

    st.divider()
    st.header("Inputs")
    uploaded_files = st.file_uploader("Upload one or more project CSVs", type=["csv"], accept_multiple_files=True)
    use_demo = st.checkbox("Use demo data", value=False)
    sims_portfolio = st.slider("Portfolio Monte Carlo sims", 250, 5000, 750, step=250)
    sims_project = st.slider("Selected project sims", 500, 20000, 1500, step=500)
    today = st.date_input("Today", value=date.today())

    st.divider()
    st.header("Manager Thresholds")
    task_green_lt = st.slider("Task Green if risk < (%)", 0, 100, 30, step=1)
    task_orange_lt = st.slider("Task Orange if risk < (%)", 0, 100, 60, step=1)
    proj_green_lt = st.slider("Project Green if delay < (%)", 0, 100, 25, step=1)
    proj_orange_lt = st.slider("Project Orange if delay < (%)", 0, 100, 50, step=1)

    st.divider()
    st.header("Cost of Delay ($)")
    default_cost_per_day = st.number_input("Default cost per day ($)", min_value=0.0, value=2500.0, step=500.0)

    st.divider()
    st.header("Portfolio Filters")
    portfolio_status_filter = st.selectbox("Show projects", ["All", "Green", "Orange", "Red"], index=0)
    portfolio_sort = st.selectbox("Sort projects by", ["Expected Delay Cost (desc)", "Delay % (desc)", "Max Task Risk % (desc)", "Project (A→Z)"], index=0)
    min_top_task_risk = st.slider("Top risks: minimum task risk (%)", 0, 100, 50, step=5)

    st.divider()
    st.header("Trends History")
    history_upload = st.file_uploader("Upload history CSV", type=["csv"], key="history_upload")
    if history_upload is not None:
        history_load_from_uploaded(history_upload)
        st.success("History loaded.")
    st.download_button("Download history CSV", data=history_to_csv_bytes(), file_name="apra_history.csv", mime="text/csv")

# Normalize thresholds
task_green = float(min(task_green_lt, task_orange_lt))
task_orange = float(max(task_green_lt, task_orange_lt))
proj_green = float(min(proj_green_lt, proj_orange_lt))
proj_orange = float(max(proj_green_lt, proj_orange_lt))

# =========================
# Portfolio load
# =========================

projects = {}
if use_demo:
    projects["Demo Project"] = pd.read_csv("sample_tasks.csv")
elif uploaded_files and len(uploaded_files) > 0:
    for f in uploaded_files:
        projects[f.name] = pd.read_csv(f)
else:
    st.info("No files uploaded — using demo data")
    projects["Demo Project"] = pd.read_csv("sample_tasks.csv")

for name, df_tmp in list(projects.items()):
    validate_csv(df_tmp)
    df_tmp = ensure_pert_columns(df_tmp)
    df_tmp = ensure_owner_columns(df_tmp)
    projects[name] = df_tmp

# =========================
# Build portfolio summary + open-items tables (Step 5)
# =========================

summary_rows = []
top_risk_tasks = []
all_open_items = []
project_analysis_cache = {}

for name, df_tmp in projects.items():
    df_an, cpath, graph = analyze_project_from_df(df_tmp, today)

    planned, samples = monte_carlo_project_duration(df_an, graph, sims=sims_portfolio)
    summ = summarize_samples(planned, samples)
    delay_pct = float(summ["delay_probability_pct"])

    max_task_risk = float(df_an["Propagated Risk %"].max())
    avg_task_risk = float(df_an["Propagated Risk %"].mean())

    proj_stat = project_status(delay_pct, proj_green, proj_orange)

    cost_day = get_cost_per_day(name, default_cost_per_day)
    exp_days = expected_delay_days(summ["planned_days"], summ["p80_days"])
    exp_cost = expected_delay_cost(delay_pct, exp_days, cost_day)

    summary_rows.append({
        "Status": proj_stat,
        "Bucket": status_bucket(proj_stat),
        "Project": name,
        "Planned (days)": summ["planned_days"],
        "P50": summ["p50_days"],
        "P80": summ["p80_days"],
        "Delay %": delay_pct,
        "Max Task Risk %": round(max_task_risk, 2),
        "Avg Task Risk %": round(avg_task_risk, 2),
        "Critical Path Tasks": len(cpath),
        "PERT?": "Yes" if any_row_has_pert_values(df_tmp) else "No",
        "Cost/day ($)": round(cost_day, 2),
        "Expected Delay Days": round(exp_days, 2),
        "Expected Delay Cost ($)": round(exp_cost, 2),
    })

    open_df = build_open_items_by_owner(
        project_name=name,
        df_tasks=df_an,
        proj_delay_pct=delay_pct,
        proj_expected_cost=exp_cost,
        task_green=task_green,
        task_orange=task_orange,
    )
    if not open_df.empty:
        all_open_items.append(open_df)

    dtop = df_an.sort_values("Propagated Risk %", ascending=False).head(3)
    for _, r in dtop.iterrows():
        tstat = task_status(float(r["Propagated Risk %"]), task_green, task_orange)
        top_risk_tasks.append({
            "Status": tstat,
            "Project": name,
            "Task": r["Task"],
            "Propagated Risk %": float(r["Propagated Risk %"]),
            "Progress %": float(r["Progress"]),
        })

    project_analysis_cache[name] = {
        "df": df_an,
        "critical_path": cpath,
        "graph": graph,
        "summary": summ,
        "delay_pct": delay_pct,
        "expected_cost": exp_cost,
    }

portfolio_df = pd.DataFrame(summary_rows)
if not portfolio_df.empty:
    if portfolio_sort == "Expected Delay Cost (desc)":
        portfolio_df = portfolio_df.sort_values("Expected Delay Cost ($)", ascending=False)
    elif portfolio_sort == "Delay % (desc)":
        portfolio_df = portfolio_df.sort_values("Delay %", ascending=False)
    elif portfolio_sort == "Max Task Risk % (desc)":
        portfolio_df = portfolio_df.sort_values("Max Task Risk %", ascending=False)
    else:
        portfolio_df = portfolio_df.sort_values("Project", ascending=True)

if portfolio_status_filter != "All" and not portfolio_df.empty:
    portfolio_df = portfolio_df[portfolio_df["Bucket"] == portfolio_status_filter]

portfolio_view_df = portfolio_df.drop(columns=["Bucket"], errors="ignore")

top_tasks_df = pd.DataFrame(top_risk_tasks)
if not top_tasks_df.empty:
    top_tasks_df = top_tasks_df[top_tasks_df["Propagated Risk %"] >= float(min_top_task_risk)]
    top_tasks_df = top_tasks_df.sort_values("Propagated Risk %", ascending=False).head(10)

trend_df = compute_trends(st.session_state["apra_history"], portfolio_view_df)

open_items_all = pd.concat(all_open_items, ignore_index=True) if all_open_items else pd.DataFrame()

# =========================
# View: Portfolio
# =========================

if view_mode == "Portfolio":
    st.subheader("Portfolio Summary")

    if not portfolio_view_df.empty:
        total_expected_cost = float(portfolio_view_df["Expected Delay Cost ($)"].sum())
        c1, c2 = st.columns(2)
        c1.metric("Total Expected Delay Cost ($)", f"{total_expected_cost:,.2f}")
        c2.metric("Projects shown", int(len(portfolio_view_df)))

    cA, cB = st.columns([1, 3])
    with cA:
        if st.button("Add snapshot (Today)"):
            history_add_snapshot(today, portfolio_view_df)
            st.success(f"Snapshot added for {today.isoformat()}.")

    if portfolio_view_df.empty:
        st.info("No projects match the current filter.")
    else:
        st.dataframe(style_status_col(portfolio_view_df, "Status"), use_container_width=True)

        st.subheader("Cost Overrides (per project)")
        proj_names = portfolio_view_df["Project"].tolist()
        sel_proj = st.selectbox("Project to override cost/day", proj_names, index=0)
        new_cost = st.number_input("Override cost/day ($)", min_value=0.0, value=float(get_cost_per_day(sel_proj, default_cost_per_day)), step=500.0)
        if st.button("Apply cost override"):
            set_cost_override(sel_proj, float(new_cost))
            st.success(f"Cost/day override set for {sel_proj}.")

    st.subheader("Top Cost Risks")
    if not portfolio_view_df.empty:
        cost_top = portfolio_view_df.sort_values("Expected Delay Cost ($)", ascending=False).head(10)
        st.dataframe(cost_top[["Status","Project","Delay %","Expected Delay Days","Cost/day ($)","Expected Delay Cost ($)"]], use_container_width=True)

    st.subheader("Trend Summary (Drift + New Reds)")
    if trend_df.empty:
        st.info("Add at least one snapshot to enable trends.")
    else:
        st.dataframe(trend_df, use_container_width=True)
        new_reds = trend_df[trend_df["New Red?"] == "YES"]
        if not new_reds.empty:
            st.warning(f"New Reds detected: {', '.join(new_reds['Project'].tolist())}")

    st.subheader("Top Risks Across Portfolio (Top 10 tasks)")
    if top_tasks_df.empty:
        st.info("No tasks match the current minimum risk filter.")
    else:
        st.dataframe(style_status_col(top_tasks_df, "Status"), use_container_width=True)

    st.subheader("History (Raw)")
    st.dataframe(st.session_state["apra_history"], use_container_width=True)

# =========================
# View: Project details (Steps 1–5)
# =========================

elif view_mode == "Project details":
    st.subheader("Project Drill-down")
    selected_project = st.selectbox("Select a project", sorted(list(projects.keys())))
    df_in = projects[selected_project]
    project_overrides = get_project_overrides(selected_project)

    hist = st.session_state["apra_history"]
    proj_hist = hist[hist["Project"] == selected_project].copy()
    if not proj_hist.empty:
        proj_hist["Snapshot Date"] = pd.to_datetime(proj_hist["Snapshot Date"], errors="coerce")
        proj_hist = proj_hist.dropna(subset=["Snapshot Date"]).sort_values("Snapshot Date")
        st.subheader("Project Trend (Delay % over snapshots)")
        st.line_chart(proj_hist.set_index("Snapshot Date")[["Delay %"]])

    df_for_analysis = apply_overrides(df_in, project_overrides)
    df, critical_path, graph = analyze_project_from_df(df_for_analysis, today)

    with st.spinner("Running Monte Carlo for selected project..."):
        planned, samples = monte_carlo_project_duration(df, graph, sims=sims_project)

    summary = summarize_samples(planned, samples)
    delay_pct_sel = float(summary["delay_probability_pct"])

    st.markdown(f"## Project Status: {project_status(delay_pct_sel, proj_green, proj_orange)}")

    cost_day = get_cost_per_day(selected_project, default_cost_per_day)
    exp_days = expected_delay_days(summary["planned_days"], summary["p80_days"])
    exp_cost = expected_delay_cost(delay_pct_sel, exp_days, cost_day)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cost/day ($)", f"{cost_day:,.2f}")
    c2.metric("Expected Delay Days (P80−Plan)", f"{exp_days:,.2f}")
    c3.metric("Expected Delay Cost ($)", f"{exp_cost:,.2f}")

    explanation = explain_project_risk(
        df=df,
        critical_path=critical_path,
        graph=graph,
        mc_summary=summary,
        task_green_lt=task_green,
        task_orange_lt=task_orange,
        proj_green_lt=proj_green,
        proj_orange_lt=proj_orange,
    )

    st.subheader("Root-Cause Summary (Why this status?)")
    st.markdown(explanation["headline"])
    st.markdown(f"**Critical Path:** {explanation['critical_path']}")
    st.markdown("**Top risk drivers:**")
    for line in explanation["driver_bullets"]:
        st.markdown(line)

    st.subheader("Action Prioritization (Impact-ranked)")
    with st.spinner("Estimating impact of corrective actions..."):
        actions_df = recommend_actions(
            df=df,
            graph=graph,
            critical_path=critical_path,
            today=today,
            baseline_summary=summary,
            max_actions=6,
            candidate_tasks=5,
            scenario_sims=700,
        )
    if actions_df.empty:
        st.info("No actions generated.")
    else:
        st.dataframe(actions_df, use_container_width=True)

    st.subheader("Task Table (with Ownership)")
    df_disp = df.copy()
    df_disp = ensure_owner_columns(df_disp)
    df_disp["Status"] = df_disp["Propagated Risk %"].apply(lambda x: task_status(float(x), task_green, task_orange))

    display_cols = [
        "Status", "Task", "Task Owner", "Risk Owner",
        "Start", "Due", "Progress",
        "Optimistic", "MostLikely", "Pessimistic",
        "Base Risk %", "Propagated Risk %", "On Critical Path"
    ]
    st.dataframe(style_status_col(df_disp[display_cols], "Status"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.pyplot(make_task_risk_fig(df), clear_figure=True)
    with right:
        st.pyplot(make_monte_carlo_fig(samples, planned), clear_figure=True)

# =========================
# View: Owners (Step 5)
# =========================

else:
    st.subheader("Open Items by Owner")

    if open_items_all.empty:
        st.info("No open items found (no Orange/Red tasks), or no data loaded.")
        st.stop()

    owner_field = st.radio("Group by", ["Risk Owner", "Task Owner"], index=0, horizontal=True)

    owners = sorted(open_items_all[owner_field].astype(str).fillna("Unassigned").replace({"": "Unassigned"}).unique().tolist())
    selected_owner = st.selectbox("Owner", ["All"] + owners, index=0)

    severity_filter = st.multiselect("Severity", ["🟥 High risk", "🟧 Recoverable"], default=["🟥 High risk", "🟧 Recoverable"])

    d = open_items_all.copy()
    d[owner_field] = d[owner_field].astype(str).fillna("Unassigned").replace({"": "Unassigned"})
    if selected_owner != "All":
        d = d[d[owner_field] == selected_owner]
    d = d[d["Task Status"].isin(severity_filter)]

    st.subheader("Owner Summary")
    summary_by_owner = summarize_open_items_by_owner(d, owner_field=owner_field)
    if summary_by_owner.empty:
        st.info("No items match the current filters.")
    else:
        st.dataframe(summary_by_owner, use_container_width=True)

    st.subheader("Open Items Detail")
    show_cols = [
        "Project", "Task Status", "Task",
        "Task Owner", "Risk Owner",
        "Propagated Risk %", "Progress", "On Critical Path",
        "Attributed Cost ($)", "Project Delay %", "Start", "Due"
    ]
    st.dataframe(d[show_cols].sort_values(["Attributed Cost ($)"], ascending=False), use_container_width=True)
