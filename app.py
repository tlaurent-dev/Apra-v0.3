import streamlit as st
import pandas as pd
from datetime import date
import tempfile, os
from io import StringIO

from apra_core import (
    analyze_project,
    monte_carlo_project_duration,
    summarize_samples,
    make_task_risk_fig,
    make_monte_carlo_fig,
    explain_project_risk,
)

# =========================
# Status (traffic lights) + styling
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
    """
    Append current portfolio metrics into history table.
    If same (Snapshot Date, Project) already exists, replace it.
    """
    init_history()
    hist = st.session_state["apra_history"]

    # Build snapshot rows from portfolio_df (which already has computed metrics/status)
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

    # Remove duplicates for same snapshot date+project, then append
    if not hist.empty:
        key = ["Snapshot Date", "Project"]
        hist = hist[~hist.set_index(key).index.isin(snap_df.set_index(key).index)]

    hist = pd.concat([hist, snap_df], ignore_index=True)

    # Sort for readability
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
        # Basic schema check
        missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Uploaded history CSV is missing columns: {', '.join(missing)}")
            st.stop()
        st.session_state["apra_history"] = df[HISTORY_COLUMNS].copy()
    except Exception as e:
        st.error(f"Failed to load history CSV: {e}")
        st.stop()

def compute_trends(history_df: pd.DataFrame, current_portfolio_df: pd.DataFrame):
    """
    For each project, compare most recent snapshot to previous snapshot (if exists).
    Detect:
    - Delay % drift and arrow
    - New Red since previous snapshot
    """
    if history_df.empty or current_portfolio_df.empty:
        return pd.DataFrame()

    hist = history_df.copy()
    hist["Snapshot Date"] = pd.to_datetime(hist["Snapshot Date"], errors="coerce")
    hist = hist.dropna(subset=["Snapshot Date"])

    # Only consider projects currently in portfolio_df (so trends align with visible set)
    current = current_portfolio_df[["Project", "Delay %", "Status", "Max Task Risk %", "P80"]].copy()
    current.rename(columns={"Delay %": "Current Delay %", "Status": "Current Status",
                            "Max Task Risk %": "Current Max Task Risk %", "P80": "Current P80"}, inplace=True)

    rows = []
    for proj in current["Project"].tolist():
        ph = hist[hist["Project"] == proj].sort_values("Snapshot Date")
        if len(ph) < 1:
            continue

        # Latest snapshot in history
        latest = ph.iloc[-1]
        prev = ph.iloc[-2] if len(ph) >= 2 else None

        curr_delay = float(current.loc[current["Project"] == proj, "Current Delay %"].iloc[0])
        curr_status = current.loc[current["Project"] == proj, "Current Status"].iloc[0]

        # Compare current vs last snapshot (not necessarily the same "today" date)
        last_delay = float(latest["Delay %"])
        delta = curr_delay - last_delay

        if abs(delta) < 0.75:
            arrow = "→"
        elif delta > 0:
            arrow = "↑"
        else:
            arrow = "↓"

        # New red detection: became Red now, but last snapshot was not Red
        last_bucket = status_bucket(str(latest["Status"]))
        now_bucket = status_bucket(str(curr_status))
        new_red = (now_bucket == "Red" and last_bucket != "Red")

        # If we have a previous snapshot, also show last drift from prev to latest
        if prev is not None:
            prev_delay = float(prev["Delay %"])
        else:
            prev_delay = None

        rows.append({
            "Project": proj,
            "Current Status": curr_status,
            "Current Delay %": round(curr_delay, 2),
            "Last Snapshot Date": latest["Snapshot Date"].date().isoformat(),
            "Last Delay %": round(last_delay, 2),
            "Δ vs Last": round(delta, 2),
            "Trend": arrow,
            "New Red?": "YES" if new_red else "",
            "Prev Snapshot Delay %": "" if prev_delay is None else round(prev_delay, 2),
        })

    return pd.DataFrame(rows)

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
    for c in ["Optimistic", "MostLikely", "Pessimistic"]:
        if c not in df.columns:
            df[c] = ""
    return df


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

with st.sidebar:
    st.header("Navigation")
    view_mode = st.radio("View", ["Portfolio", "Project details"], index=0)

    st.divider()
    st.header("Inputs")
    uploaded_files = st.file_uploader("Upload one or more project CSVs", type=["csv"], accept_multiple_files=True)
    use_demo = st.checkbox("Use demo data", value=False)
    sims_portfolio = st.slider("Portfolio Monte Carlo sims", 250, 5000, 750, step=250)
    sims_project = st.slider("Selected project sims", 500, 20000, 1500, step=500)
    today = st.date_input("Today", value=date.today())

    st.divider()
    st.header("Manager Thresholds")

    st.caption("Task status uses Propagated Risk %.")
    task_green_lt = st.slider("Task Green if risk < (%)", 0, 100, 30, step=1)
    task_orange_lt = st.slider("Task Orange if risk < (%)", 0, 100, 60, step=1)

    st.caption("Project status uses Delay % (Monte Carlo).")
    proj_green_lt = st.slider("Project Green if delay < (%)", 0, 100, 25, step=1)
    proj_orange_lt = st.slider("Project Orange if delay < (%)", 0, 100, 50, step=1)

    if task_orange_lt < task_green_lt:
        st.warning("Task thresholds: Orange threshold should be >= Green threshold.")
    if proj_orange_lt < proj_green_lt:
        st.warning("Project thresholds: Orange threshold should be >= Green threshold.")

    st.divider()
    st.header("Portfolio Filters")
    portfolio_status_filter = st.selectbox("Show projects", ["All", "Green", "Orange", "Red"], index=0)
    portfolio_sort = st.selectbox("Sort projects by", ["Delay % (desc)", "Max Task Risk % (desc)", "Project (A→Z)"], index=0)
    min_top_task_risk = st.slider("Top risks: minimum task risk (%)", 0, 100, 50, step=5)

    st.divider()
    st.header("Trends History (Step 2)")
    st.caption("Add snapshots over time, download/upload history to preserve trends across sessions.")
    history_upload = st.file_uploader("Upload history CSV", type=["csv"], key="history_upload")
    if history_upload is not None:
        history_load_from_uploaded(history_upload)
        st.success("History loaded.")

    st.download_button(
        "Download history CSV",
        data=history_to_csv_bytes(),
        file_name="apra_history.csv",
        mime="text/csv"
    )

    if st.button("Clear history"):
        st.session_state["apra_history"] = pd.DataFrame(columns=HISTORY_COLUMNS)
        st.success("History cleared.")

# Normalize thresholds even if inverted
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
        try:
            df_tmp = pd.read_csv(f)
            projects[f.name] = df_tmp
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
            st.stop()
else:
    st.info("No files uploaded — using demo data")
    projects["Demo Project"] = pd.read_csv("sample_tasks.csv")

for name, df_tmp in list(projects.items()):
    validate_csv(df_tmp)
    projects[name] = ensure_pert_columns(df_tmp)

# =========================
# Build portfolio summary (shared)
# =========================

summary_rows = []
top_risk_tasks = []

for name, df_tmp in projects.items():
    df_an, cpath, graph = analyze_project_from_df(df_tmp, today)

    planned, samples = monte_carlo_project_duration(df_an, graph, sims=sims_portfolio)
    summ = summarize_samples(planned, samples)
    delay_pct = float(summ["delay_probability_pct"])

    max_task_risk = float(df_an["Propagated Risk %"].max())
    avg_task_risk = float(df_an["Propagated Risk %"].mean())

    proj_stat = project_status(delay_pct, proj_green, proj_orange)

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
    })

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

portfolio_df = pd.DataFrame(summary_rows)
if not portfolio_df.empty:
    if portfolio_sort == "Delay % (desc)":
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

# =========================
# Trends section (available in both views)
# =========================

trend_df = compute_trends(st.session_state["apra_history"], portfolio_view_df)

# =========================
# View: Portfolio
# =========================

if view_mode == "Portfolio":
    st.subheader("Portfolio Summary")

    cA, cB = st.columns([1, 3])
    with cA:
        if st.button("Add snapshot (Today)"):
            # Add snapshot using the unfiltered portfolio table? Use filtered view to match what user sees.
            history_add_snapshot(today, portfolio_view_df if not portfolio_view_df.empty else portfolio_df.drop(columns=["Bucket"], errors="ignore"))
            st.success(f"Snapshot added for {today.isoformat()}.")

    if portfolio_view_df.empty:
        st.info("No projects match the current filter.")
    else:
        st.dataframe(style_status_col(portfolio_view_df, "Status"), use_container_width=True)

    st.subheader("Trend Summary (Drift + New Reds)")
    if trend_df.empty:
        st.info("Add at least one snapshot to enable trends.")
    else:
        # Highlight New Red? column visually by keeping it simple
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
# View: Project details (includes Step 1 root-cause + Step 2 trend summary)
# =========================

else:
    st.subheader("Project Drill-down")
    selected_project = st.selectbox("Select a project", sorted(list(projects.keys())))
    df_in = projects[selected_project]
    project_overrides = get_project_overrides(selected_project)

    # Project-level trend strip (if history exists)
    hist = st.session_state["apra_history"]
    proj_hist = hist[hist["Project"] == selected_project].copy()
    if not proj_hist.empty:
        proj_hist["Snapshot Date"] = pd.to_datetime(proj_hist["Snapshot Date"], errors="coerce")
        proj_hist = proj_hist.dropna(subset=["Snapshot Date"]).sort_values("Snapshot Date")
        st.subheader("Project Trend (Delay % over snapshots)")
        chart_df = proj_hist.set_index("Snapshot Date")[["Delay %"]]
        st.line_chart(chart_df)

    st.subheader("What-if controls (Selected project)")
    st.caption("Override Optimistic / MostLikely / Pessimistic (and Progress) for a task. This does not edit your CSV file.")

    colA, colB = st.columns([2, 1])
    task_list = df_in["Task"].astype(str).tolist()
    selected_task = colA.selectbox("Select task to override", task_list, index=0 if task_list else None)

    with colB:
        if st.button("Reset overrides (this project)"):
            st.session_state["overrides_by_project"][selected_project] = {}
            project_overrides = get_project_overrides(selected_project)
            st.success("Overrides cleared.")

    if selected_task:
        row = df_in.loc[df_in["Task"].astype(str) == str(selected_task)].iloc[0]

        def _val(x, default):
            try:
                if pd.isna(x) or str(x).strip() == "":
                    return default
                return float(x)
            except Exception:
                return default

        cur_o = _val(row.get("Optimistic", ""), 1.0)
        cur_m = _val(row.get("MostLikely", ""), max(cur_o, 2.0))
        cur_p = _val(row.get("Pessimistic", ""), max(cur_m, 3.0))
        cur_prog = float(row.get("Progress", 0))

        st.markdown(f"**Selected task:** `{selected_task}`")

        c1, c2, c3, c4 = st.columns(4)
        new_o = c1.number_input("Optimistic (days)", min_value=0.5, value=float(cur_o), step=0.5, key=f"{selected_project}:{selected_task}:o")
        new_m = c2.number_input("Most Likely (days)", min_value=0.5, value=float(cur_m), step=0.5, key=f"{selected_project}:{selected_task}:m")
        new_p = c3.number_input("Pessimistic (days)", min_value=0.5, value=float(cur_p), step=0.5, key=f"{selected_project}:{selected_task}:p")
        new_prog = c4.slider("Progress (%)", 0, 100, int(round(cur_prog)), step=1, key=f"{selected_project}:{selected_task}:prog")

        if not (new_o <= new_m <= new_p):
            st.error("Invalid PERT override: must satisfy Optimistic ≤ MostLikely ≤ Pessimistic.")
        else:
            colS1, colS2 = st.columns([1, 2])
            if colS1.button("Apply override", key=f"{selected_project}:{selected_task}:apply"):
                project_overrides[str(selected_task)] = {
                    "Optimistic": float(new_o),
                    "MostLikely": float(new_m),
                    "Pessimistic": float(new_p),
                    "Progress": float(new_prog),
                }
                st.success("Override applied.")

            if colS2.button("Remove override", key=f"{selected_project}:{selected_task}:remove"):
                project_overrides.pop(str(selected_task), None)
                st.success("Override removed.")

    # Run analysis with overrides applied
    df_for_analysis = apply_overrides(df_in, project_overrides)
    df, critical_path, graph = analyze_project_from_df(df_for_analysis, today)

    with st.spinner("Running Monte Carlo for selected project..."):
        planned, samples = monte_carlo_project_duration(df, graph, sims=sims_project)

    summary = summarize_samples(planned, samples)
    delay_pct_sel = float(summary["delay_probability_pct"])

    st.markdown(f"## Project Status: {project_status(delay_pct_sel, proj_green, proj_orange)}")

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
    st.caption(explanation["meaning"])

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Planned (days)", summary["planned_days"])
    c2.metric("P50", summary["p50_days"])
    c3.metric("P80", summary["p80_days"])
    c4.metric("Delay %", delay_pct_sel)

    st.subheader("Task Table")
    df_disp = df.copy()
    df_disp["Status"] = df_disp["Propagated Risk %"].apply(lambda x: task_status(float(x), task_green, task_orange))

    display_cols = [
        "Status", "Task", "Start", "Due", "Progress",
        "Optimistic", "MostLikely", "Pessimistic",
        "Base Risk %", "Propagated Risk %", "On Critical Path"
    ]
    st.dataframe(style_status_col(df_disp[display_cols], "Status"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.pyplot(make_task_risk_fig(df), clear_figure=True)
    with right:
        st.pyplot(make_monte_carlo_fig(samples, planned), clear_figure=True)

    st.subheader("Active overrides (selected project)")
    if project_overrides:
        st.json(project_overrides)
    else:
        st.caption("No overrides applied.")
