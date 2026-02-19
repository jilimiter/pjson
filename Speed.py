# Speed.py / app.py
# - Time(랩 내부) + LapNumber로 연속 시간축(cum_sec) 생성
# - Start/Stop/Reset 자동 재생 (st.fragment 기반)
# - Speed HUD(게이지/메트릭) + Speed 타임라인(빨간 점, 보기 좋은 x축)
# - 2D 트랙(빨간 점) 실시간 표시:
#     * CSV에 X,Y가 있으면 X_plot/Y_plot로 보간(interpolate)해서 끊김 최소화
#     * 그래도 NaN이면 마지막 유효 위치(last_xy)로 점 유지

import os
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Speed HUD (Auto Play + 2D Track)", layout="wide")


# -------------------------
# Data processing
# -------------------------
def build_cumulative_time_from_time(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Time", "LapNumber", "Speed"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 없습니다.")

    df = df.copy()

    # Time -> seconds
    df["Time_td"] = pd.to_timedelta(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time_td"]).copy()
    df["t_sec"] = df["Time_td"].dt.total_seconds()

    # LapNumber -> int
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df = df.dropna(subset=["LapNumber"]).copy()
    df["LapNumber"] = df["LapNumber"].astype(int)

    # Speed numeric
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").ffill().fillna(0)

    # Optional columns
    for c in ["RPM", "nGear", "Throttle", "Distance", "RelativeDistance", "X", "Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["RPM"] = df["RPM"].ffill().fillna(0) if "RPM" in df.columns else 0
    df["nGear"] = df["nGear"].ffill().fillna(0).astype(int) if "nGear" in df.columns else 0

    # Sort within lap
    df = df.sort_values(["LapNumber", "t_sec"]).reset_index(drop=True)

    # Build cumulative time
    lap_dur = df.groupby("LapNumber")["t_sec"].max().sort_index()
    lap_offset = lap_dur.shift(1).fillna(0).cumsum()
    df["lap_offset"] = df["LapNumber"].map(lap_offset).astype(float)
    df["cum_sec"] = df["lap_offset"] + df["t_sec"]

    # Final sort by cum time
    df = df.sort_values("cum_sec").reset_index(drop=True)

    # ---- 2D position smoothing (핵심) ----
    # X/Y가 있으면 보간해서 X_plot/Y_plot 생성 (끊김 최소화)
    if ("X" in df.columns) and ("Y" in df.columns):
        df["X_plot"] = df["X"].interpolate(limit_direction="both")
        df["Y_plot"] = df["Y"].interpolate(limit_direction="both")
    else:
        df["X_plot"] = np.nan
        df["Y_plot"] = np.nan

    return df


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(BytesIO(file_bytes))
    return build_cumulative_time_from_time(raw)


# -------------------------
# Plot helpers
# -------------------------
def speed_gauge(speed: float, vmax: float) -> go.Figure:
    v = float(np.clip(speed, 0, vmax))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": " km/h"},
            gauge={"axis": {"range": [0, vmax]}, "bar": {"thickness": 0.25}},
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=35, b=10))
    return fig


def make_timeline_figure(df: pd.DataFrame, row: pd.Series, tick_interval_s: int = 300) -> go.Figure:
    current_x = float(row["cum_sec"])
    current_y = float(row["Speed"])

    x_max = float(df["cum_sec"].max())
    tickvals = np.arange(0, x_max + 1e-9, int(tick_interval_s))
    ticktext = [f"{int(v//60)}:{int(v%60):02d}" for v in tickvals]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["cum_sec"],
            y=df["Speed"],
            mode="lines",
            name="Speed",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current_x],
            y=[current_y],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Current",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(
        title="Time [mm:ss]",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    fig.update_yaxes(title="Speed [km/h]")
    return fig


def make_track_2d_figure(df: pd.DataFrame, row: pd.Series) -> go.Figure:
    # plot용 컬럼이 없으면 안내
    if ("X_plot" not in df.columns) or ("Y_plot" not in df.columns):
        fig = go.Figure()
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[
                dict(
                    text="X/Y 좌표 컬럼이 없어 2D 트랙을 표시할 수 없습니다.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    track_xy = df[["X_plot", "Y_plot"]].dropna()

    cur_x = row.get("X_plot", np.nan)
    cur_y = row.get("Y_plot", np.nan)

    # 현재가 NaN이면 마지막 유효 위치 유지
    if "last_xy" not in st.session_state:
        st.session_state.last_xy = None

    if pd.notna(cur_x) and pd.notna(cur_y):
        st.session_state.last_xy = (float(cur_x), float(cur_y))

    fig = go.Figure()

    # 트랙 라인
    fig.add_trace(
        go.Scatter(
            x=track_xy["X_plot"],
            y=track_xy["Y_plot"],
            mode="lines",
            line=dict(width=3),
            name="Track",
        )
    )

    # 차량 점(마지막 유효 위치라도 표시)
    if st.session_state.last_xy is not None:
        lx, ly = st.session_state.last_xy
        fig.add_trace(
            go.Scatter(
                x=[lx],
                y=[ly],
                mode="markers",
                marker=dict(color="red", size=12),
                name="Car",
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


# -------------------------
# Streamlit version guard (st.fragment needed)
# -------------------------
if not hasattr(st, "fragment"):
    st.error(
        "현재 Streamlit 버전에서 st.fragment(run_every=...)가 없습니다.\n"
        "Streamlit을 최신 버전으로 업데이트해야 자동 재생이 안정적으로 동작합니다."
    )
    st.stop()


# -------------------------
# UI
# -------------------------
st.title("Speed HUD (Auto Playback + 2D Track)")

with st.sidebar:
    st.subheader("설정")
    vmax = st.number_input("게이지 최고속 (km/h)", 100, 450, 360, 10)
    fps = st.selectbox("FPS", [5, 10, 20, 30], index=1)
    step_rows = st.selectbox("프레임당 행 이동", [1, 2, 5, 10], index=0)
    tick_interval_s = st.selectbox("x축 간격", [60, 120, 300, 600], index=2)  # 1/2/5/10분
    show_debug = st.checkbox("debug 표시", value=False)

    if st.button("캐시 초기화"):
        st.cache_data.clear()
        st.session_state.pop("file_bytes", None)
        st.session_state.pop("file_name", None)
        st.session_state.pop("last_xy", None)
        st.rerun()

uploaded = st.file_uploader("Telemetry CSV 업로드", type=["csv"], key="uploader")
if uploaded is not None:
    st.session_state.file_bytes = uploaded.getvalue()
    st.session_state.file_name = uploaded.name

if "file_bytes" not in st.session_state:
    st.info("CSV를 업로드해줘.")
    st.stop()

df = load_and_process(st.session_state.file_bytes)

st.caption(f"RUNNING FILE: {os.path.abspath(__file__)}")
st.caption(f"FILE: {st.session_state.get('file_name','')} | ROWS: {len(df):,}")
st.caption(f"cum_sec range: {df['cum_sec'].min():.3f} ~ {df['cum_sec'].max():.3f}")
st.caption(f"X/Y available: {('X' in df.columns) and ('Y' in df.columns)}")

# -------------------------
# Playback state
# -------------------------
if "playing" not in st.session_state:
    st.session_state.playing = False
if "i_now" not in st.session_state:
    st.session_state.i_now = 0

b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
with b1:
    if st.button("Start", use_container_width=True):
        st.session_state.playing = True
with b2:
    if st.button("Stop", use_container_width=True):
        st.session_state.playing = False
with b3:
    if st.button("Reset", use_container_width=True):
        st.session_state.playing = False
        st.session_state.i_now = 0
        st.session_state.last_xy = None
with b4:
    st.write(f"상태: {'Playing' if st.session_state.playing else 'Paused'}")

# paused에서만 seek
if not st.session_state.playing:
    st.session_state.i_now = st.slider(
        "Index Seek",
        0,
        len(df) - 1,
        int(st.session_state.i_now),
        1,
        key="seek_idx",
    )

# -------------------------
# Layout containers (고정 위치)
# -------------------------
hud_container = st.container()

# 아래 영역: 왼쪽(속도 그래프) / 오른쪽(2D 트랙)
col_left, col_right = st.columns([2, 1])
with col_left:
    timeline_container = st.container()
with col_right:
    track_container = st.container()

debug_container = st.container()

with hud_container:
    hud_ph = st.empty()
with timeline_container:
    timeline_ph = st.empty()
with track_container:
    track_ph = st.empty()
with debug_container:
    debug_ph = st.empty()

# -------------------------
# Auto tick fragment
# -------------------------
@st.fragment(run_every=1.0 / float(fps))
def tick():
    # advance index
    if st.session_state.playing:
        st.session_state.i_now = int(st.session_state.i_now) + int(step_rows)
        if st.session_state.i_now >= len(df):
            st.session_state.i_now = len(df) - 1
            st.session_state.playing = False

    # current row
    i = int(np.clip(st.session_state.i_now, 0, len(df) - 1))
    row = df.iloc[i]

    # debug
    if show_debug:
        with debug_container:
            debug_ph.write(
                {
                    "i_now": i,
                    "cum_sec": float(row["cum_sec"]),
                    "speed": float(row["Speed"]),
                    "x_plot": None if pd.isna(row.get("X_plot", np.nan)) else float(row["X_plot"]),
                    "y_plot": None if pd.isna(row.get("Y_plot", np.nan)) else float(row["Y_plot"]),
                    "playing": bool(st.session_state.playing),
                }
            )
    else:
        with debug_container:
            debug_ph.empty()

    # HUD
    with hud_container:
        c1, c2, c3 = hud_ph.columns([2, 1, 1])
        with c1:
            st.plotly_chart(speed_gauge(float(row["Speed"]), vmax=vmax), use_container_width=True)
        with c2:
            st.metric("Cum Time [s]", f"{float(row['cum_sec']):.2f}")
            st.metric("Lap", f"{int(row['LapNumber'])}")
            st.metric("Lap Time [s]", f"{float(row['t_sec']):.2f}")
        with c3:
            st.metric("Speed [km/h]", f"{float(row['Speed']):.1f}")
            st.metric("RPM", f"{float(row.get('RPM', 0)):.0f}")
            st.metric("Gear", f"{int(row.get('nGear', 0))}")

    # Speed timeline (빨간 점 + 보기 좋은 x축)
    fig_tl = make_timeline_figure(df, row, tick_interval_s=int(tick_interval_s))
    with timeline_container:
        timeline_ph.plotly_chart(fig_tl, use_container_width=True)

    # 2D Track (보간 + 마지막 유효 위치 유지)
    fig_tr = make_track_2d_figure(df, row)
    with track_container:
        track_ph.plotly_chart(fig_tr, use_container_width=True)

# run fragment
tick()