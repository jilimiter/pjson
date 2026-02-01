# app.py
# Streamlit + FastF1 + Plotly Animation:
# 2D track with moving marker + rolling-window (high-res) Brake/Throttle view
#
# Run:
#   streamlit run app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import fastf1
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="F1 Track + Rolling Telemetry", layout="wide")
st.title("🏁 F1 Track 2D + Rolling Brake/Throttle (High-Res)")
st.caption("오른쪽은 전체 랩이 아니라 현재 시점 주변 'rolling window'만 확대해서 보여줍니다. (라인 재그리기 X, x축 range만 이동)")

# Cache
CACHE_DIR = os.path.join(os.getcwd(), "fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Sidebar
st.sidebar.header("Controls")

with st.sidebar.form("controls"):
    year = st.selectbox("Year", [2023, 2024, 2025], index=0)
    event_key = st.text_input("Event key", value="Japan").strip()
    session_code = st.selectbox("Session", ["R", "Q", "S", "FP1", "FP2", "FP3"], index=0)
    driver = st.text_input("Driver code", value="VER").upper().strip()

    st.markdown("---")
    st.markdown("### Sampling / Playback")
    resample_dt = st.slider("Resample dt (sec)", 0.02, 0.50, 0.05, 0.01)
    frame_step = st.slider("Advance per frame (points)", 1, 30, 5, 1)
    fps = st.slider("Playback FPS", 5, 60, 20, 1)

    st.markdown("---")
    st.markdown("### Rolling window (sec)")
    window_len = st.slider("Window length (sec)", 3.0, 30.0, 12.0, 0.5)
    window_mode = st.selectbox("Window mode", ["Trailing (past only)", "Centered (past+future)"], index=0)

    st.markdown("---")
    show_corner_split = st.checkbox("Show corner/straight split on track", value=True)
    curvature_threshold = st.slider("Curvature threshold", 0.001, 0.020, 0.004, 0.001)

    run_btn = st.form_submit_button("Load & Build Replay")

@st.cache_data(show_spinner=True)
def load_fastest_lap(year: int, event_key: str, session_code: str, driver: str):
    sess = fastf1.get_session(year, event_key, session_code)
    sess.load()

    laps = sess.laps.pick_driver(driver)
    if laps.empty:
        raise ValueError(f"No laps for driver '{driver}' in {year} {event_key} {session_code}")

    lap = laps.pick_fastest()
    tel = lap.get_telemetry().add_distance()

    required = {"Time", "X", "Y", "Brake", "Throttle"}
    missing = required - set(tel.columns)
    if missing:
        raise ValueError(f"Telemetry missing columns: {missing}")

    df = pd.DataFrame({
        "t": tel["Time"].dt.total_seconds(),
        "x": tel["X"],
        "y": tel["Y"],
        "brake": tel["Brake"].astype(int) * 100.0,
        "throttle_raw": tel["Throttle"].astype(float),
    }).dropna()

    df = df.sort_values("t").drop_duplicates("t")

    th = df["throttle_raw"].to_numpy(dtype=float)
    if np.nanmax(th) <= 1.5:
        th = th * 100.0
    df["throttle"] = np.clip(th, 0, 100)
    df = df.drop(columns=["throttle_raw"])

    meta = {
        "Year": year,
        "EventName": sess.event.get("EventName", ""),
        "Location": sess.event.get("Location", ""),
        "Session": session_code,
        "Driver": driver,
        "LapTime": str(lap["LapTime"]) if "LapTime" in lap else "",
        "LapNumber": int(lap["LapNumber"]) if "LapNumber" in lap else None,
    }
    return df, meta

def resample_time(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    t0, t1 = float(df["t"].min()), float(df["t"].max())
    tg = np.arange(t0, t1, dt)

    def interp(col):
        return np.interp(tg, df["t"].to_numpy(), df[col].to_numpy())

    return pd.DataFrame({
        "t": tg,
        "x": interp("x"),
        "y": interp("y"),
        "brake": interp("brake"),
        "throttle": interp("throttle"),
    })

def compute_curvature_flags(x: np.ndarray, y: np.ndarray, thr: float):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx*dx + dy*dy) + 1e-9
    theta = np.unwrap(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    kappa = np.abs(dtheta) / ds[1:]
    kappa = np.r_[kappa[0], kappa, kappa[-1]]
    return kappa > thr

def build_fig(df_rs: pd.DataFrame, meta: dict, frame_step: int, fps: int,
              window_len: float, window_mode: str,
              show_corner_split: bool, curvature_threshold: float):

    t = df_rs["t"].to_numpy(dtype=float)
    x = df_rs["x"].to_numpy(dtype=float)
    y = df_rs["y"].to_numpy(dtype=float)
    brake = df_rs["brake"].to_numpy(dtype=float)
    throttle = df_rs["throttle"].to_numpy(dtype=float)

    # track split
    if show_corner_split and len(x) > 10:
        is_corner = compute_curvature_flags(x, y, curvature_threshold)
        x_corner = x.copy(); y_corner = y.copy()
        x_str = x.copy(); y_str = y.copy()
        x_corner[~is_corner] = np.nan; y_corner[~is_corner] = np.nan
        x_str[is_corner] = np.nan; y_str[is_corner] = np.nan
    else:
        x_corner = np.full_like(x, np.nan); y_corner = np.full_like(y, np.nan)
        x_str = x; y_str = y

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=("2D Track (X-Y) + Position", "Rolling Window (Brake/Throttle)"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # Track lines
    fig.add_trace(go.Scatter(x=x_str, y=y_str, mode="lines", name="Track (straight-ish)", connectgaps=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_corner, y=y_corner, mode="lines", name="Track (corner-ish)", connectgaps=False), row=1, col=1)

    # Moving marker (trace idx 2)
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="Car", marker=dict(size=10)), row=1, col=1)

    # FULL signals ONCE (static) — high-res comes from moving x-range (camera), not replot
    fig.add_trace(go.Scatter(x=t, y=brake, mode="lines", name="Brake", line=dict(dash="solid")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=throttle, mode="lines", name="Throttle", line=dict(dash="dash")), row=1, col=2)

    # Cursor line (trace idx 5)
    fig.add_trace(go.Scatter(x=[t[0], t[0]], y=[0, 100], mode="lines", name="Cursor", line=dict(dash="dot")), row=1, col=2)

    # axes
    fig.update_xaxes(title_text="X", row=1, col=1, showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="Y", row=1, col=1, showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="0..100", row=1, col=2, showgrid=True, range=[-5, 105])

    title = f"{meta.get('Year')} {meta.get('EventName')} {meta.get('Session')} — {meta.get('Driver')} (Fastest Lap {meta.get('LapTime')})"
    fig.update_layout(
        title=title,
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # window function
    def window_range(ti: float):
        if window_mode.startswith("Trailing"):
            lo = ti - window_len
            hi = ti
        else:  # Centered
            lo = ti - window_len/2
            hi = ti + window_len/2
        lo = max(float(t[0]), lo)
        hi = min(float(t[-1]), hi)
        # ensure non-zero width
        if hi - lo < 1e-6:
            hi = min(float(t[-1]), lo + 0.5)
        return [lo, hi]

    # frames: update marker + cursor + xaxis2 range only (no line redraw)
    frames = []
    idxs = list(range(0, len(t), frame_step))
    if idxs[-1] != len(t) - 1:
        idxs.append(len(t) - 1)

    for i in idxs:
        ti = float(t[i])
        xr = window_range(ti)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[x[i]], y=[y[i]]),           # trace 2
                    go.Scatter(x=[ti, ti], y=[0, 100]),       # trace 5
                ],
                traces=[2, 5],
                layout=go.Layout(
                    xaxis2=dict(range=xr)
                ),
                name=str(i),
            )
        )

    fig.frames = frames

    # animation controls
    frame_duration_ms = int(1000 / max(1, fps))

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.05,
                y=1.12,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=frame_duration_ms, redraw=False),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode="immediate"
                        )],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.05, y=1.02, len=0.90,
                steps=[
                    dict(
                        method="animate",
                        args=[[fr.name], dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0)
                        )],
                        label=f"{int(fr.name)}"
                    )
                    for fr in frames
                ],
                currentvalue=dict(prefix="Frame idx: "),
            )
        ],
    )

    # set initial window
    fig.update_layout(xaxis2=dict(range=window_range(float(t[0]))))

    return fig

# Main
if not run_btn:
    st.info("⬅️ 왼쪽에서 설정 후 **Load & Build Replay**를 누르세요.")
    st.stop()

try:
    df_raw, meta = load_fastest_lap(int(year), event_key, session_code, driver)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

df_rs = resample_time(df_raw, float(resample_dt))

st.subheader("Loaded lap info")
st.dataframe(pd.DataFrame([meta]), use_container_width=True)

st.markdown("---")

fig = build_fig(
    df_rs=df_rs,
    meta=meta,
    frame_step=int(frame_step),
    fps=int(fps),
    window_len=float(window_len),
    window_mode=str(window_mode),
    show_corner_split=bool(show_corner_split),
    curvature_threshold=float(curvature_threshold),
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "오른쪽은 rolling window로 확대되어 보입니다. "
    "재생이 끝난 후에도 Plotly 줌/팬으로 더 확대 분석 가능합니다."
)
