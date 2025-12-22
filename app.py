import streamlit as st
import pandas as pd
from datetime import date, datetime
import tempfile, os
import io

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
# UI Settings (SAFE - Step 7)
# =========================
# Design goal: add settings without touching business logic
# - No new dependencies
# - No new navigation view required
# - Uses CSS injection only (Streamlit-safe)
# - Defaults persist for the session via st.session_state

def init_ui_settings():
    st.session_state.setdefault("ui_theme", "Professional Light")
    st.session_state.setdefault("ui_font", "System")
    st.session_state.setdefault("ui_density", "Comfortable")

    # Safety: if an old session stored a removed theme, reset it
    if st.session_state.get("ui_theme") not in THEMES:
        st.session_state["ui_theme"] = "Professional Light"

THEMES = {
    "Professional Light": {"bg": "#ffffff", "panel": "#f7f7f9", "text": "#111827", "grid": "#e5e7eb"},
    "Minimal Gray": {"bg": "#f3f4f6", "panel": "#ffffff", "text": "#111827", "grid": "#d1d5db"},
}

FONTS = {
    "System": 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
    "Georgia": 'Georgia, "Times New Roman", Times, serif',
    "Monospace": 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
}

DENSITY = {
    "Comfortable": {"font": "0.95rem", "pad": "0.75rem"},
    "Compact": {"font": "0.90rem", "pad": "0.35rem"},
}

def apply_ui_theme():
    """
    Production-safe UI theming via CSS injection.

    Design principles:
    - Avoid f-strings inside CSS to prevent brace-related syntax issues.
    - Use explicit placeholder replacement (no Template dependency).
    - Keep business logic untouched: styling only.
    - Ensure sidebar + BaseWeb widgets remain readable in all themes.
    """
    t = THEMES.get(st.session_state.get("ui_theme", "Professional Light"), THEMES["Professional Light"])
    font = FONTS.get(st.session_state.get("ui_font", "System"), FONTS["System"])
    d = DENSITY.get(st.session_state.get("ui_density", "Comfortable"), DENSITY["Comfortable"])

    css = r"""
    <style>
      :root {
        --apra-bg: __APRA_BG__;
        --apra-panel: __APRA_PANEL__;
        --apra-text: __APRA_TEXT__;
        --apra-grid: __APRA_GRID__;
        --apra-font-size: __APRA_FONT_SIZE__;
        --apra-pad: __APRA_PAD__;
        --apra-font-family: __APRA_FONT_FAMILY__;
      }

      html, body, [class*="stApp"] {
        background: var(--apra-bg) !important;
        color: var(--apra-text) !important;
        font-family: var(--apra-font-family) !important;
        font-size: var(--apra-font-size) !important;
      }

      /* Sidebar container */
      [data-testid="stSidebar"] {
        background: var(--apra-panel) !important;
        border-right: 1px solid var(--apra-grid) !important;
      }

      /* ===============================
         Sidebar readability (theme-safe)
         =============================== */

      /* Ensure common sidebar text nodes are readable (does NOT force backgrounds) */
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] h4,
      [data-testid="stSidebar"] h5,
      [data-testid="stSidebar"] h6,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] small,
      [data-testid="stSidebar"] label {
        color: var(--apra-text) !important;
        opacity: 1 !important;
      }

      [data-testid="stSidebar"] hr {
        border-color: var(--apra-grid) !important;
        opacity: 0.9 !important;
      }

      /* ===============================
         Widget readability (Streamlit/BaseWeb)
         =============================== */

      /* BaseWeb select / input shells in the sidebar */
      [data-testid="stSidebar"] div[data-baseweb="select"] > div,
      [data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background-color: var(--apra-panel) !important;
        border: 1px solid var(--apra-grid) !important;
      }

      /* Selected value + input text (sidebar) */
      [data-testid="stSidebar"] div[data-baseweb="select"] span,
      [data-testid="stSidebar"] div[data-baseweb="select"] input,
      [data-testid="stSidebar"] div[data-baseweb="input"] input {
        color: var(--apra-text) !important;
        -webkit-text-fill-color: var(--apra-text) !important;
        caret-color: var(--apra-text) !important;
      }

      /* Dropdown popover/menu (applies app-wide) */
      div[data-baseweb="popover"],
      div[data-baseweb="menu"] {
        background-color: var(--apra-panel) !important;
        border: 1px solid var(--apra-grid) !important;
      }

      div[data-baseweb="menu"] *,
      div[role="listbox"] *,
      div[role="option"] {
        color: var(--apra-text) !important;
        opacity: 1 !important;
      }

      /* Placeholder text */
      [data-testid="stSidebar"] input::placeholder,
      [data-testid="stSidebar"] textarea::placeholder,
      [data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder {
        color: rgba(127, 127, 127, 0.95) !important;
        opacity: 1 !important;
      }

      /* Metric cards */
      [data-testid="stMetric"] {
        background: var(--apra-panel) !important;
        border: 1px solid var(--apra-grid) !important;
        border-radius: 14px !important;
        padding: var(--apra-pad) !important;
      }

      /* Buttons */
      .stButton > button {
        background: var(--apra-panel) !important;
        color: var(--apra-text) !important;
        border-radius: 10px !important;
        border: 1px solid var(--apra-grid) !important;
        padding: calc(var(--apra-pad) * 0.7) var(--apra-pad) !important;
      }

      /* Inputs (general) */
      input, textarea {
        border-radius: 10px !important;
        color: var(--apra-text) !important;
      }

      /* Links */
      a, a:visited { color: #60a5fa !important; }

      /* Text blocks */
      .stMarkdown, .stText, .stCaption, .stMarkdown p, .stMarkdown li {
        color: var(--apra-text) !important;
      }
      .stCaption { opacity: 0.88; }

      /* Dataframe container (outer frame) */
      [data-testid="stDataFrame"] {
        background: var(--apra-panel) !important;
        border: 1px solid var(--apra-grid) !important;
        border-radius: 14px !important;
        padding: calc(var(--apra-pad) * 0.6) !important;
      }

      /* Expanders */
      [data-testid="stExpander"] {
        background: var(--apra-panel) !important;
        border: 1px solid var(--apra-grid) !important;
        border-radius: 14px !important;
      }
    </style>
    """

    # Safe placeholder substitution (no brace escaping issues)
    css = (css
           .replace("__APRA_BG__", str(t["bg"]))
           .replace("__APRA_PANEL__", str(t["panel"]))
           .replace("__APRA_TEXT__", str(t["text"]))
           .replace("__APRA_GRID__", str(t["grid"]))
           .replace("__APRA_FONT_SIZE__", str(d["font"]))
           .replace("__APRA_PAD__", str(d["pad"]))
           .replace("__APRA_FONT_FAMILY__", str(font)))

    st.markdown(css, unsafe_allow_html=True)

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

    # Enforce 2-decimal formatting for numeric columns at render time (display only).
    sty = df.style.applymap(_style, subset=[col_name])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        sty = sty.format({c: "{:,.2f}" for c in num_cols})
    return sty

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
    d = df_tasks.copy()
    d["Propagated Risk %"] = d["Propagated Risk %"].astype(float)
    d["Progress"] = d["Progress"].astype(float)
    d = ensure_owner_columns(d)

    d["Task Status"] = d["Propagated Risk %"].apply(lambda x: task_status(float(x), task_green, task_orange))
    open_mask = d["Task Status"].isin(["🟧 Recoverable", "🟥 High risk"])
    open_df = d.loc[open_mask].copy()

    if open_df.empty:
        return open_df

    risk_sum = float(open_df["Propagated Risk %"].sum())
    if risk_sum <= 0:
        open_df["Attributed Cost ($)"] = 0.0
    else:
        open_df["Attributed Cost ($)"] = open_df["Propagated Risk %"].apply(
            lambda r: (float(r) / risk_sum) * float(proj_expected_cost)
        )

    # FIX: avoid duplicate "Project" column
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
    return g.sort_values(["Attributed_Cost_USD", "Red_Items"], ascending=[False, False])

# =========================
# Weekly Auto-Reports (Step 6)
# =========================

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _pdf_make_table(df: pd.DataFrame, max_rows: int = 35):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    if df is None or df.empty:
        data = [["(no data)"]]
        t = Table(data)
        t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
        return t

    d = df.copy().head(max_rows)
    data = [list(d.columns)] + d.astype(str).values.tolist()

    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
    ]))
    return t

def make_owner_weekly_pdf(run_dt: datetime, owner_field: str, owner_summary: pd.DataFrame, open_items: pd.DataFrame) -> bytes:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet

    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=letter, title="APRA Weekly Owner Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("APRA — Weekly Owner Report", styles["Title"]))
    story.append(Paragraph(f"Run date: {run_dt.strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Grouping: {owner_field}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Owner Summary (Open Items)", styles["Heading2"]))
    story.append(_pdf_make_table(owner_summary, max_rows=30))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Open Items Detail (Top by Attributed Cost)", styles["Heading2"]))
    detail = open_items.sort_values("Attributed Cost ($)", ascending=False) if not open_items.empty else open_items
    story.append(_pdf_make_table(detail, max_rows=35))

    doc.build(story)
    return buff.getvalue()

def make_project_weekly_pdf(run_dt: datetime, portfolio_df: pd.DataFrame, top_cost_df: pd.DataFrame, trends_df: pd.DataFrame) -> bytes:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet

    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=letter, title="APRA Weekly Project Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("APRA — Weekly Project Report", styles["Title"]))
    story.append(Paragraph(f"Run date: {run_dt.strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Portfolio Summary", styles["Heading2"]))
    story.append(_pdf_make_table(portfolio_df, max_rows=25))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Cost Risks", styles["Heading2"]))
    story.append(_pdf_make_table(top_cost_df, max_rows=15))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Trend Summary (Drift + New Reds)", styles["Heading2"]))
    story.append(_pdf_make_table(trends_df, max_rows=20))

    doc.build(story)
    return buff.getvalue()

# =========================
# Display rounding (no logic changes)
# =========================

def round_numeric_cols(df: pd.DataFrame, cols=None, n: int = 2) -> pd.DataFrame:
    """
    Round numeric columns for display/export only.
    This function must NOT be used on inputs to core calculations.
    """
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if cols is None:
        cols_to_round = list(out.select_dtypes(include="number").columns)
    else:
        cols_to_round = [c for c in cols if c in out.columns]
    for c in cols_to_round:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(n)
    return out

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
init_ui_settings()
apply_ui_theme()

st.title("APRA — Project Risk Dashboard")

init_overrides()
init_history()
init_cost_overrides()

with st.sidebar:
    st.header("Navigation")
    view_mode = st.radio("View", ["Portfolio", "Project details", "Owners", "Reports"], index=0)

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

    # SAFE UI settings panel (does not affect computations)
    st.divider()
    st.subheader("UI Settings")
    st.selectbox("Theme", list(THEMES.keys()), key="ui_theme")
    st.selectbox("Font", list(FONTS.keys()), key="ui_font")
    st.selectbox("Density", list(DENSITY.keys()), key="ui_density")
    apply_ui_theme()

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
# Build portfolio summary + open-items tables
# =========================

summary_rows = []
top_risk_tasks = []
all_open_items = []

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

# Display-only rounding (does not affect computations)
PORTFOLIO_NUM_COLS = [
    "Planned (days)", "P50", "P80", "Delay %",
    "Max Task Risk %", "Avg Task Risk %",
    "Cost/day ($)", "Expected Delay Days", "Expected Delay Cost ($)"
]
TOP_TASKS_NUM_COLS = ["Propagated Risk %", "Progress %"]
TREND_NUM_COLS = ["Current Delay %", "Last Delay %", "Δ vs Last"]
OPEN_ITEMS_NUM_COLS = ["Project Delay %", "Propagated Risk %", "Base Risk %", "Progress", "Attributed Cost ($)"]

portfolio_view_disp = round_numeric_cols(portfolio_view_df, PORTFOLIO_NUM_COLS, n=2) if isinstance(portfolio_view_df, pd.DataFrame) else portfolio_view_df
top_tasks_disp = round_numeric_cols(top_tasks_df, TOP_TASKS_NUM_COLS, n=2) if isinstance(top_tasks_df, pd.DataFrame) else top_tasks_df
trend_disp = round_numeric_cols(trend_df, TREND_NUM_COLS, n=2) if isinstance(trend_df, pd.DataFrame) else trend_df
open_items_disp = round_numeric_cols(open_items_all, OPEN_ITEMS_NUM_COLS, n=2) if isinstance(open_items_all, pd.DataFrame) else open_items_all

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

    cA, _ = st.columns([1, 3])
    with cA:
        if st.button("Add snapshot (Today)"):
            history_add_snapshot(today, portfolio_view_df)
            st.success(f"Snapshot added for {today.isoformat()}.")

    if portfolio_view_df.empty:
        st.info("No projects match the current filter.")
    else:
        st.dataframe(style_status_col(portfolio_view_disp, "Status"), use_container_width=True)

        st.subheader("Cost Overrides (per project)")
        proj_names = portfolio_view_df["Project"].tolist()
        sel_proj = st.selectbox("Project to override cost/day", proj_names, index=0)
        new_cost = st.number_input("Override cost/day ($)", min_value=0.0, value=float(get_cost_per_day(sel_proj, default_cost_per_day)), step=500.0)
        if st.button("Apply cost override"):
            set_cost_override(sel_proj, float(new_cost))
            st.success(f"Cost/day override set for {sel_proj}.")

    st.subheader("Top Cost Risks")
    if not portfolio_view_df.empty:
        cost_top = portfolio_view_disp.sort_values("Expected Delay Cost ($)", ascending=False).head(10)
        st.dataframe(cost_top[["Status","Project","Delay %","Expected Delay Days","Cost/day ($)","Expected Delay Cost ($)"]], use_container_width=True)

    st.subheader("Trend Summary (Drift + New Reds)")
    if trend_df.empty:
        st.info("Add at least one snapshot to enable trends.")
    else:
        # Ensure numeric Δ exists and is rounded for display
        td = trend_disp.copy()
        if "Δ vs Last" in td.columns:
            td["Δ vs Last"] = pd.to_numeric(td["Δ vs Last"], errors="coerce")

        st.dataframe(td.style.format(precision=2), use_container_width=True)

        # --- KPI tiles ---
        new_reds_df = td[td["New Red?"] == "YES"].copy()
        worsening = td.sort_values("Δ vs Last", ascending=False).head(5).copy()
        improving = td.sort_values("Δ vs Last", ascending=True).head(5).copy()

        largest_worsen = float(worsening["Δ vs Last"].iloc[0]) if (not worsening.empty and pd.notna(worsening["Δ vs Last"].iloc[0])) else 0.0
        largest_improve = float(improving["Δ vs Last"].iloc[0]) if (not improving.empty and pd.notna(improving["Δ vs Last"].iloc[0])) else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("New Reds", int(len(new_reds_df)))
        c2.metric("Largest Worsening Δ", f"{largest_worsen:,.2f}")
        c3.metric("Largest Improvement Δ", f"{largest_improve:,.2f}")
        c4.metric("Projects with trends", int(len(td)))

        # --- New Reds (Escalate) ---
        st.markdown("### New Reds (Escalate now)")
        if new_reds_df.empty:
            st.caption("No new Reds since the last snapshot.")
        else:
            st.warning("Escalate these projects: " + ", ".join(new_reds_df["Project"].astype(str).tolist()))
            show_cols = ["Project", "Current Status", "Current Delay %", "Last Snapshot Date", "Last Delay %", "Δ vs Last", "Trend"]
            for c in show_cols:
                if c not in new_reds_df.columns:
                    new_reds_df[c] = ""
            st.dataframe(new_reds_df[show_cols].style.format(precision=2), use_container_width=True)

        # --- Biggest Worsening / Improving ---
        st.markdown("### Biggest Drift (since last snapshot)")
        left, right = st.columns(2)

        with left:
            st.markdown("**Top Worsening**")
            if worsening.empty:
                st.caption("No trend data.")
            else:
                show_cols = ["Project", "Current Status", "Current Delay %", "Last Delay %", "Δ vs Last", "Trend"]
                for c in show_cols:
                    if c not in worsening.columns:
                        worsening[c] = ""
                st.dataframe(worsening[show_cols].style.format(precision=2), use_container_width=True)

        with right:
            st.markdown("**Top Improving**")
            if improving.empty:
                st.caption("No trend data.")
            else:
                show_cols = ["Project", "Current Status", "Current Delay %", "Last Delay %", "Δ vs Last", "Trend"]
                for c in show_cols:
                    if c not in improving.columns:
                        improving[c] = ""
                st.dataframe(improving[show_cols].style.format(precision=2), use_container_width=True)

    st.subheader("Top Risks Across Portfolio (Top 10 tasks)")
    if top_tasks_df.empty:
        st.info("No tasks match the current minimum risk filter.")
    else:
        st.dataframe(style_status_col(top_tasks_disp, "Status"), use_container_width=True)

# =========================
# View: Project details
# =========================

elif view_mode == "Project details":
    st.subheader("Project Drill-down")
    selected_project = st.selectbox("Select a project", sorted(list(projects.keys())))
    df_in = projects[selected_project]
    project_overrides = get_project_overrides(selected_project)

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
        st.dataframe(round_numeric_cols(actions_df, cols=None, n=2).style.format(precision=2), use_container_width=True)

    st.subheader("Task Table (with Ownership)")
    df_disp = ensure_owner_columns(df.copy())
    df_disp = round_numeric_cols(df_disp, cols=["Progress","Base Risk %","Propagated Risk %"], n=2)
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
# View: Owners
# =========================

elif view_mode == "Owners":
    st.subheader("Open Items by Owner")

    if open_items_all.empty:
        st.info("No open items found (no Orange/Red tasks), or no data loaded.")
        st.stop()

    owner_field = st.radio("Group by", ["Risk Owner", "Task Owner"], index=0, horizontal=True)

    owners = sorted(open_items_disp[owner_field].astype(str).fillna("Unassigned").replace({"": "Unassigned"}).unique().tolist())
    selected_owner = st.selectbox("Owner", ["All"] + owners, index=0)
    severity_filter = st.multiselect("Severity", ["🟥 High risk", "🟧 Recoverable"], default=["🟥 High risk", "🟧 Recoverable"])

    d = open_items_disp.copy()
    d[owner_field] = d[owner_field].astype(str).fillna("Unassigned").replace({"": "Unassigned"})
    if selected_owner != "All":
        d = d[d[owner_field] == selected_owner]
    d = d[d["Task Status"].isin(severity_filter)]

    st.subheader("Owner Summary")
    summary_by_owner = summarize_open_items_by_owner(d, owner_field=owner_field)
    summary_by_owner = round_numeric_cols(summary_by_owner, cols=["Attributed_Cost_USD","Avg_Task_Risk_Pct"], n=2)
    if summary_by_owner.empty:
        st.info("No items match the current filters.")
    else:
        st.dataframe(summary_by_owner.style.format(precision=2), use_container_width=True)

    st.subheader("Open Items Detail")
    show_cols = [
        "Project", "Task Status", "Task",
        "Task Owner", "Risk Owner",
        "Propagated Risk %", "Progress", "On Critical Path",
        "Attributed Cost ($)", "Project Delay %", "Start", "Due"
    ]
    st.dataframe(d[show_cols].sort_values(["Attributed Cost ($)"], ascending=False).style.format(precision=2), use_container_width=True)

# =========================
# View: Reports (Step 6)
# =========================

else:
    st.subheader("Weekly Auto-Reports (PDF / CSV)")

    run_dt = datetime.now()

    tab1, tab2 = st.tabs(["By Owner", "By Project"])

    with tab1:
        if open_items_all.empty:
            st.info("No open items to report (no Orange/Red tasks).")
        else:
            owner_field = st.radio("Group by", ["Risk Owner", "Task Owner"], index=0, horizontal=True, key="rep_owner_group")
            severity_filter = st.multiselect("Include severities", ["🟥 High risk", "🟧 Recoverable"], default=["🟥 High risk", "🟧 Recoverable"], key="rep_owner_sev")

            d = open_items_disp.copy()
            d = d[d["Task Status"].isin(severity_filter)]

            owner_summary = summarize_open_items_by_owner(d, owner_field=owner_field)
            # Display/export rounding only
            owner_summary = round_numeric_cols(owner_summary, cols=["Attributed_Cost_USD","Avg_Task_Risk_Pct"], n=2)
            d_detail = round_numeric_cols(d.sort_values("Attributed Cost ($)", ascending=False), cols=OPEN_ITEMS_NUM_COLS, n=2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Open items", int(len(d_detail)))
            c2.metric("Total attributed cost ($)", f"{float(d_detail['Attributed Cost ($)'].sum()):,.2f}" if not d_detail.empty else "0.00")
            c3.metric("Owners", int(owner_summary["Owner"].nunique()) if not owner_summary.empty else 0)

            st.dataframe(owner_summary.style.format(precision=2), use_container_width=True)
            st.dataframe(d_detail.style.format(precision=2), use_container_width=True)

            st.download_button(
                "Download Owner Summary CSV",
                data=df_to_csv_bytes(owner_summary),
                file_name=f"apra_owner_summary_{run_dt.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download Open Items CSV",
                data=df_to_csv_bytes(d_detail),
                file_name=f"apra_open_items_{run_dt.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download Owner Report PDF",
                data=make_owner_weekly_pdf(run_dt, owner_field, owner_summary, d_detail),
                file_name=f"apra_owner_report_{run_dt.strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    with tab2:
        if portfolio_view_df.empty:
            st.info("No portfolio data loaded.")
        else:
            portfolio_out = portfolio_view_disp.copy()
            top_cost = round_numeric_cols(portfolio_out.sort_values("Expected Delay Cost ($)", ascending=False).head(10), cols=PORTFOLIO_NUM_COLS, n=2)
            trends_out = trend_disp.copy() if isinstance(trend_disp, pd.DataFrame) else pd.DataFrame()

            total_expected_cost = float(portfolio_out["Expected Delay Cost ($)"].sum()) if "Expected Delay Cost ($)" in portfolio_out.columns else 0.0
            c1, c2 = st.columns(2)
            c1.metric("Projects", int(len(portfolio_out)))
            c2.metric("Total Expected Delay Cost ($)", f"{total_expected_cost:,.2f}")

            st.subheader("Portfolio Summary (filtered)")
            st.dataframe(style_status_col(portfolio_out, "Status"), use_container_width=True)

            st.subheader("Top Cost Risks")
            st.dataframe(top_cost[["Status","Project","Delay %","Expected Delay Days","Cost/day ($)","Expected Delay Cost ($)"]], use_container_width=True)

            st.subheader("Trend Summary")
            if trends_out.empty:
                st.caption("No trends available (add at least one snapshot).")
            else:
                st.dataframe(trends_out.style.format(precision=2), use_container_width=True)

            st.download_button(
                "Download Portfolio CSV",
                data=df_to_csv_bytes(portfolio_out),
                file_name=f"apra_portfolio_{run_dt.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download Project Report PDF",
                data=make_project_weekly_pdf(run_dt, portfolio_out, top_cost, trends_out),
                file_name=f"apra_project_report_{run_dt.strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
