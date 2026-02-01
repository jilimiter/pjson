# liveBrakeThrottleMap.py
# Streamlit + Uploaded Telemetry (CSV/Excel) + Plotly Animation
# Default: Overlay comparison
# Optional: Side-by-side view (tabs)
#
# Run:
#   streamlit run liveBrakeThrottleMap.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Live Brake/Throttle Comparison", layout="wide")
st.title("🏁 Live Brake/Throttle Map (Upload & Compare)")
st.caption("기본은 overlay(겹쳐보기) 비교입니다. 옵션을 켜면 side-by-side(나란히)도 볼 수 있어요.")

REQUIRED_COLS = ["Time", "Throttle", "Brake", "X", "Y"]

# -----------------------------
# Helpers
# -----------------------------
def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().any():
        return td.dt.total_seconds()

    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().any():
        return (dt - dt.iloc[0]).dt.total_seconds()

    return pd.Series(np.nan, index=series.index)


def load_telemetry(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("지원 형식이 아닙니다. CSV 또는 XLSX 파일을 업로드하세요.")

    df.columns = [c.strip() for c in df.columns]
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(list(missing))}\n필수: {REQUIRED_COLS}")

    df = df.copy()

    # Brake -> 0..100
    if df["Brake"].dtype == bool:
        brake = df["Brake"].astype(int) * 100.0
    else:
        if df["Brake"].dtype == object:
            b = df["Brake"].astype(str).str.lower()
            brake = np.where(b.isin(["true", "1", "yes"]), 100.0,
                             np.where(b.isin(["false", "0", "no"]), 0.0, np.nan))
            brake = pd.Series(brake, index=df.index)
        else:
            brake = df["Brake"].astype(float)
            if brake.max(skipna=True) <= 1.5:
                brake = brake * 100.0
    df["brake"] = np.clip(brake.astype(float), 0, 100)

    # Throttle -> 0..100
    thr = df["Throttle"].astype(float)
    if thr.max(skipna=True) <= 1.5:
        thr = thr * 100.0
    df["throttle"] = np.clip(thr, 0, 100)

    # Time axes
    df["t_lap"] = _to_seconds(df["Time"])
    df["t_session"] = _to_seconds(df["SessionTime"]) if "SessionTime" in df.columns else np.nan

    # Coords
    df["x"] = pd.to_numeric(df["X"], errors="coerce")
    df["y"] = pd.to_numeric(df["Y"], errors="coerce")

    # Distance (optional but recommended for comparison)
    df["dist"] = pd.to_numeric(df["Distance"], errors="coerce") if "Distance" in df.columns else np.nan

    # LapNumber + Status
    df["lap"] = pd.to_numeric(df["LapNumber"], errors="coerce") if "LapNumber" in df.columns else np.nan
    df["status"] = df["Status"].astype(str) if "Status" in df.columns else ""

    # Clean
    df = df.dropna(subset=["t_lap", "x", "y", "brake", "throttle"]).copy()
    return df


def _ensure_monotonic(df: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[axis_col]).sort_values(axis_col).copy()
    df = df[df[axis_col].diff().fillna(0) >= 0].copy()
    return df


def build_common_grid(dfs: list[pd.DataFrame], axis_col: str, step: float) -> np.ndarray:
    mins = [float(d[axis_col].dropna().min()) for d in dfs]
    maxs = [float(d[axis_col].dropna().max()) for d in dfs]
    a0 = max(mins)
    a1 = min(maxs)
    if a1 <= a0:
        raise ValueError("데이터들 사이에 공통 비교 구간이 없습니다. (axis 범위가 겹치지 않음)")
    grid = np.arange(a0, a1, step)
    if len(grid) < 20:
        raise ValueError("공통 구간이 너무 짧습니다. (grid 포인트가 부족)")
    return grid


def resample_on_grid(df: pd.DataFrame, axis_col: str, grid: np.ndarray) -> pd.DataFrame:
    a = df[axis_col].to_numpy(dtype=float)

    def interp(col):
        return np.interp(grid, a, df[col].to_numpy(dtype=float))

    return pd.DataFrame({
        "a": grid,               # axis (time or distance)
        "x": interp("x"),
        "y": interp("y"),
        "brake": interp("brake"),
        "throttle": interp("throttle"),
    })


def build_fig_single(df_rs: pd.DataFrame, axis_label: str,
                     fps: int, frame_step: int, window_len: float, window_mode: str) -> go.Figure:
    a = df_rs["a"].to_numpy(dtype=float)
    x = df_rs["x"].to_numpy(dtype=float)
    y = df_rs["y"].to_numpy(dtype=float)
    brake = df_rs["brake"].to_numpy(dtype=float)
    throttle = df_rs["throttle"].to_numpy(dtype=float)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=("2D Track (X-Y) + Position", f"Rolling Window (Brake/Throttle) — {axis_label}"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Track"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="Car", marker=dict(size=10)), row=1, col=1)

    fig.add_trace(go.Scatter(x=a, y=brake, mode="lines", name="Brake"), row=1, col=2)
    fig.add_trace(go.Scatter(x=a, y=throttle, mode="lines", name="Throttle", line=dict(dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[a[0], a[0]], y=[0, 100], mode="lines", name="Cursor", line=dict(dash="dot")), row=1, col=2)

    fig.update_xaxes(title_text="X", row=1, col=1, showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="Y", row=1, col=1, showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text=axis_label, row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="0..100", row=1, col=2, showgrid=True, range=[-5, 105])

    fig.update_layout(height=650, hovermode="x unified", legend=dict(orientation="h"),
                      margin=dict(l=20, r=20, t=60, b=20))

    def window_range(ai: float):
        if window_mode.startswith("Trailing"):
            lo, hi = ai - window_len, ai
        else:
            lo, hi = ai - window_len / 2, ai + window_len / 2
        lo = max(float(a[0]), lo)
        hi = min(float(a[-1]), hi)
        if hi - lo < 1e-6:
            hi = min(float(a[-1]), lo + window_len * 0.3)
        return [lo, hi]

    frames = []
    idxs = list(range(0, len(a), frame_step))
    if idxs[-1] != len(a) - 1:
        idxs.append(len(a) - 1)

    # trace index: car marker=1, cursor=4
    for i in idxs:
        ai = float(a[i])
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[x[i]], y=[y[i]]),           # marker
                    go.Scatter(x=[ai, ai], y=[0, 100]),       # cursor
                ],
                traces=[1, 4],
                layout=go.Layout(xaxis2=dict(range=window_range(ai))),
                name=str(i),
            )
        )

    fig.frames = frames
    frame_duration_ms = int(1000 / max(1, fps))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.05, y=1.12,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=False),
                                      transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            x=0.05, y=1.02, len=0.90,
            steps=[dict(method="animate",
                        args=[[fr.name], dict(mode="immediate",
                                              frame=dict(duration=0, redraw=False),
                                              transition=dict(duration=0))],
                        label=f"{int(fr.name)}") for fr in frames],
            currentvalue=dict(prefix="Frame idx: "),
        )],
    )

    fig.update_layout(xaxis2=dict(range=window_range(float(a[0]))))
    return fig


def build_fig_overlay(datasets_rs: list[tuple[str, pd.DataFrame]], axis_label: str,
                      fps: int, frame_step: int, window_len: float, window_mode: str) -> go.Figure:
    # assume all share same grid "a"
    a = datasets_rs[0][1]["a"].to_numpy(dtype=float)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=("2D Track (X-Y) + Multi Position", f"Overlay Brake/Throttle — Rolling Window ({axis_label})"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    marker_trace_idxs = []

    # For each dataset: track line + marker + brake + throttle
    for name, df_rs in datasets_rs:
        x = df_rs["x"].to_numpy(dtype=float)
        y = df_rs["y"].to_numpy(dtype=float)
        brake = df_rs["brake"].to_numpy(dtype=float)
        throttle = df_rs["throttle"].to_numpy(dtype=float)

        # Track
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{name} Track"), row=1, col=1)
        # Marker
        fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name=f"{name} Car", marker=dict(size=10)),
                      row=1, col=1)
        marker_trace_idxs.append(len(fig.data) - 1)

        # Signals
        fig.add_trace(go.Scatter(x=a, y=brake, mode="lines", name=f"{name} Brake"), row=1, col=2)
        fig.add_trace(go.Scatter(x=a, y=throttle, mode="lines", name=f"{name} Throttle", line=dict(dash="dash")),
                      row=1, col=2)

    # One shared cursor line at end
    fig.add_trace(go.Scatter(x=[a[0], a[0]], y=[0, 100], mode="lines",
                             name="Cursor", line=dict(dash="dot")), row=1, col=2)
    cursor_trace_idx = len(fig.data) - 1

    fig.update_xaxes(title_text="X", row=1, col=1, showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="Y", row=1, col=1, showgrid=True, zeroline=False,
                     scaleanchor="x", scaleratio=1)

    fig.update_xaxes(title_text=axis_label, row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="0..100", row=1, col=2, showgrid=True, range=[-5, 105])

    fig.update_layout(height=650, hovermode="x unified", legend=dict(orientation="h"),
                      margin=dict(l=20, r=20, t=60, b=20))

    def window_range(ai: float):
        if window_mode.startswith("Trailing"):
            lo, hi = ai - window_len, ai
        else:
            lo, hi = ai - window_len / 2, ai + window_len / 2
        lo = max(float(a[0]), lo)
        hi = min(float(a[-1]), hi)
        if hi - lo < 1e-6:
            hi = min(float(a[-1]), lo + window_len * 0.3)
        return [lo, hi]

    frames = []
    idxs = list(range(0, len(a), frame_step))
    if idxs[-1] != len(a) - 1:
        idxs.append(len(a) - 1)

    # Update all markers + one cursor line
    for i in idxs:
        ai = float(a[i])
        frame_data = []
        frame_traces = []

        # markers
        for (name, df_rs), m_idx in zip(datasets_rs, marker_trace_idxs):
            x = float(df_rs["x"].iloc[i])
            y = float(df_rs["y"].iloc[i])
            frame_data.append(go.Scatter(x=[x], y=[y]))
            frame_traces.append(m_idx)

        # cursor
        frame_data.append(go.Scatter(x=[ai, ai], y=[0, 100]))
        frame_traces.append(cursor_trace_idx)

        frames.append(go.Frame(
            data=frame_data,
            traces=frame_traces,
            layout=go.Layout(xaxis2=dict(range=window_range(ai))),
            name=str(i),
        ))

    fig.frames = frames
    frame_duration_ms = int(1000 / max(1, fps))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.05, y=1.12,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=False),
                                      transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            x=0.05, y=1.02, len=0.90,
            steps=[dict(method="animate",
                        args=[[fr.name], dict(mode="immediate",
                                              frame=dict(duration=0, redraw=False),
                                              transition=dict(duration=0))],
                        label=f"{int(fr.name)}") for fr in frames],
            currentvalue=dict(prefix="Frame idx: "),
        )],
    )

    fig.update_layout(xaxis2=dict(range=window_range(float(a[0]))))
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Telemetry Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload telemetry files (.csv / .xlsx) — multiple allowed",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.header("Compare mode")
show_side_by_side = st.sidebar.toggle("Show side-by-side view", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Axis + Playback")
compare_axis = st.sidebar.selectbox("Compare axis", ["Distance (m)", "Time (lap s)", "SessionTime (session s)"], index=0)

# dt for time, step for distance
if compare_axis.startswith("Distance"):
    grid_step = st.sidebar.slider("Grid step (m)", 0.5, 10.0, 1.0, 0.5)
else:
    grid_step = st.sidebar.slider("Grid step (sec)", 0.02, 0.50, 0.05, 0.01)

frame_step = st.sidebar.slider("Advance per frame (points)", 1, 30, 5, 1)
fps = st.sidebar.slider("Playback FPS", 5, 60, 20, 1)

st.sidebar.markdown("---")
st.sidebar.header("Rolling window")
window_len = st.sidebar.slider("Window length (axis units)", 3.0, 60.0, 12.0, 0.5)
window_mode = st.sidebar.selectbox("Window mode", ["Trailing (past only)", "Centered (past+future)"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Filters")
only_ontrack = st.sidebar.checkbox("Keep only Status == OnTrack (if exists)", value=True)

run_btn = st.sidebar.button("Build Comparison")


# -----------------------------
# Main
# -----------------------------
if not uploaded_files or len(uploaded_files) < 2:
    st.info("왼쪽에서 Telemetry 파일을 **2개 이상** 업로드하세요. (예: 2023/2024/2025)")
    st.stop()

# Load all
datasets = []
load_errors = []
for f in uploaded_files:
    try:
        df = load_telemetry(f)
        if only_ontrack and df["status"].astype(str).str.len().gt(0).any():
            df = df[df["status"].str.lower() == "ontrack"].copy()
        datasets.append((f.name, df))
    except Exception as e:
        load_errors.append((f.name, str(e)))

if load_errors:
    st.error("일부 파일 로드 실패:")
    for name, msg in load_errors:
        st.write(f"- {name}: {msg}")
    st.stop()

# Determine axis column
if compare_axis.startswith("Distance"):
    axis_col = "dist"
    axis_label = "Distance (m)"
elif compare_axis.startswith("SessionTime"):
    axis_col = "t_session"
    axis_label = "SessionTime (s)"
else:
    axis_col = "t_lap"
    axis_label = "Time (lap s)"

# Guard for missing dist/session
for name, df in datasets:
    if axis_col not in df.columns or df[axis_col].isna().all():
        st.error(f"[{name}]에서 '{axis_label}' 축을 만들 수 없습니다. (컬럼 누락/파싱 실패)")
        st.stop()

# Optional: common LapNumber selection if possible
all_have_lap = all(df["lap"].notna().any() for _, df in datasets)
if all_have_lap:
    lap_sets = []
    for _, df in datasets:
        lap_sets.append(set(int(x) for x in df["lap"].dropna().unique()))
    common_laps = sorted(list(set.intersection(*lap_sets))) if lap_sets else []
else:
    common_laps = []

lap_sel = None
if common_laps:
    lap_sel = st.sidebar.selectbox("Common LapNumber (optional)", ["(All)"] + [str(x) for x in common_laps], index=0)
    if lap_sel != "(All)":
        lap_sel = int(lap_sel)

# Apply lap filter if chosen
if lap_sel is not None and lap_sel != "(All)":
    datasets = [(name, df[df["lap"] == float(lap_sel)].copy()) for name, df in datasets]

# Preview
st.subheader("Uploaded datasets preview (head)")
for name, df in datasets:
    st.markdown(f"**{name}**  (rows={len(df)})")
    st.dataframe(df.head(8), use_container_width=True)

if not run_btn:
    st.info("⬅️ 설정 후 **Build Comparison** 버튼을 누르세요.")
    st.stop()

# Prepare monotonic axis + common grid
dfs_axis = []
for name, df in datasets:
    d = _ensure_monotonic(df, axis_col)
    if len(d) < 50:
        st.error(f"[{name}] 유효 데이터가 너무 적습니다. 필터/축을 확인하세요.")
        st.stop()
    dfs_axis.append(d)

try:
    grid = build_common_grid(dfs_axis, axis_col=axis_col, step=float(grid_step))
except Exception as e:
    st.error(str(e))
    st.stop()

# Resample each dataset onto common grid
datasets_rs = []
for (name, _), d in zip(datasets, dfs_axis):
    rs = resample_on_grid(d, axis_col=axis_col, grid=grid)
    datasets_rs.append((name, rs))

st.markdown("---")
st.subheader("Overlay comparison (default)")
fig_overlay = build_fig_overlay(
    datasets_rs=datasets_rs,
    axis_label=axis_label,
    fps=int(fps),
    frame_step=int(frame_step),
    window_len=float(window_len),
    window_mode=str(window_mode),
)
st.plotly_chart(fig_overlay, use_container_width=True)

if show_side_by_side:
    st.markdown("---")
    st.subheader("Side-by-side view")
    tabs = st.tabs([name for name, _ in datasets_rs])

    for (name, rs), tab in zip(datasets_rs, tabs):
        with tab:
            fig_single = build_fig_single(
                df_rs=rs,
                axis_label=axis_label,
                fps=int(fps),
                frame_step=int(frame_step),
                window_len=float(window_len),
                window_mode=str(window_mode),
            )
            st.plotly_chart(fig_single, use_container_width=True)

st.info("팁: Distance 축으로 비교하면(권장) 같은 트랙 위치에서 Brake/Throttle 차이를 직관적으로 볼 수 있습니다.")
