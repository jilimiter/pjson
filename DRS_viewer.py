#DRS_viewer.py
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
import streamlit.components.v1 as components
import plotly.graph_objects as go



# =========================================================
# Config / Schema
# =========================================================
REQUIRED_COLS_DRS = ["Time", "DRS"]


@dataclass(frozen=True)
class AnimConfig:
    max_frames: int = 3000
    target_fps: int = 60
    min_frame_duration_ms: int = 16


@dataclass(frozen=True)
class DrsChartConfig:
    height: int = 260
    y_min: float = -0.1
    y_max: float = 1.1


# =========================================================
# Data Loading / Parsing
# =========================================================
def read_csv(uploaded_file) -> pd.DataFrame:
    """Streamlit UploadedFile -> raw DataFrame"""
    content = uploaded_file.read()
    return pd.read_csv(io.BytesIO(content))


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> Optional[str]:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return f"CSV에 필요한 컬럼이 없습니다: {missing}\n필요 컬럼: {required_cols}"
    return None


def parse_time_to_timedelta(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """'0 days 00:00:01.144000' 같은 문자열을 timedelta로 변환"""
    out = df.copy()
    out[time_col] = pd.to_timedelta(out[time_col], errors="coerce")
    return out


def normalize_drs_to_01(series: pd.Series) -> np.ndarray:
    """
    DRS 값을 0/1 ndarray로 정규화.
    지원: bool, int/float, str(True/False/0/1)
    """
    s = series

    # pandas BooleanDtype / bool
    if s.dtype == bool or str(s.dtype).lower() == "boolean":
        arr = s.astype(int).to_numpy()

    # object(문자열 섞임 가능)
    elif s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        if ss.isin(["true", "false"]).all():
            arr = (ss == "true").astype(int).to_numpy()
        else:
            arr = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()

    # numeric
    else:
        arr = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()

    return np.clip(arr, 0, 1)


def load_drs_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """
    FastF1 export CSV에서 DRS용 최소 데이터프레임만 반환.
    반환 컬럼:
      - Time (timedelta)
      - DRS (0/1 int)
      - time_s (float seconds, UI용)
    """
    try:
        df = read_csv(uploaded_file)

        err = validate_required_columns(df, REQUIRED_COLS_DRS)
        if err:
            st.error(err)
            return None

        df = parse_time_to_timedelta(df, "Time")
        if df["Time"].isna().all():
            st.error("Time 컬럼을 timedelta로 파싱할 수 없습니다.")
            return None

        df = df.sort_values("Time").reset_index(drop=True)

        drs01 = normalize_drs_to_01(df["DRS"])

        out = pd.DataFrame(
            {
                "Time": df["Time"],
                "time_s": df["Time"].dt.total_seconds().astype(float),
                "DRS": drs01.astype(int),
            }
        )

        return out

    except Exception as e:
        st.error(f"파일 파싱 중 오류 발생: {e}")
        return None

def render_plotly_autoplay(fig: go.Figure, height: int = 300):
    html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=True,   # <- 핵심
    )
    components.html(html, height=height)

# =========================================================
# Animation helpers
# =========================================================
def downsample_for_animation(df: pd.DataFrame, cfg: AnimConfig):
    n = len(df)
    step = max(1, n // cfg.max_frames)
    df_s = df.iloc[::step].reset_index(drop=True)

    frame_duration_ms = max(cfg.min_frame_duration_ms, int(1000 / cfg.target_fps))
    return df_s, frame_duration_ms


# =========================================================
# Plot builders
# =========================================================
def build_drs_timeline_figure(
    df: pd.DataFrame,
    anim_cfg: AnimConfig = AnimConfig(),
    chart_cfg: DrsChartConfig = DrsChartConfig(),
    show_static_bar: bool = True,
) -> go.Figure:
    """
    DRS 타임라인 Figure 생성.
    - 정적 bar(전체 타임라인) + 애니메이션 커서(현재 시간)
    - 프레임에는 커서만 업데이트하여 메시지 크기 폭발 방지
    """
    t_full = df["time_s"].to_numpy()
    drs_full = df["DRS"].to_numpy()

    df_s, frame_duration_ms = downsample_for_animation(df, anim_cfg)
    t_s = df_s["time_s"].to_numpy()

    t0, t_end = float(t_full[0]), float(t_full[-1])
    y0, y1 = chart_cfg.y_min, chart_cfg.y_max

    fig = go.Figure()

    # bar (옵션)
    if show_static_bar:
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=drs_full,
                mode="markers",
                name="DRS",
                marker=dict(size=5, opacity=0.4),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[t0, t_end],
                y=[0, 0],
                mode="lines",
                showlegend=False,
            )
        )

    # (2) 커서(세로선) trace: 이 trace만 프레임에서 업데이트
    cursor_x = float(t_s[0]) if len(t_s) else t0
    cursor_y = float(df_s["DRS"].iloc[0]) if len(df_s) else 0

    fig.add_trace(
        go.Scatter(
            x=[cursor_x],
            y=[cursor_y],
            mode="markers",
            name="Current",
            showlegend=False,
            marker=dict(size=14, opacity=1.0, color="red",),
        )
    )

    # 프레임: 커서만 이동 (traces=[1]로 두 번째 trace만 갱신)
    frames = []
    for i in range(len(t_s)):
        cx = float(t_s[i])
        cy = float(df_s["DRS"].iloc[i])

        frames.append(
            go.Frame(
                data=[go.Scatter(x=[cx], y=[cy])],
                traces=[1],   # 두 번째 trace(Current)만 업데이트
                name=str(i),
            )
        )
    fig.frames = frames

    fig.update_layout(
        title="DRS ON/OFF Timeline",
        xaxis_title="Time [s]",
        yaxis_title="DRS (0=OFF, 1=ON)",
        yaxis=dict( range=[y0, y1],
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["OFF", "ON"],),
        xaxis=dict(range=[t0, t_end]),
        height=chart_cfg.height,
        margin=dict(t=40, b=40, l=40, r=20),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.15,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{t_s[i]:.2f}s",
                    }
                    for i in range(len(t_s))
                ],
                "x": 0.0,
                "y": -0.10,
                "len": 1.0,
            }
        ],
    )

    return fig


# =========================================================
# Streamlit UI (thin wrapper)
# =========================================================
def render_plotly(fig: go.Figure, height: int, auto_play: bool):
    """
    auto_play=True일 때 components.html로 렌더.
    CDN 막혀도 보이도록 include_plotlyjs='inline' 사용.
    실패하면 st.plotly_chart로 fallback.
    """
    if not auto_play:
        st.plotly_chart(fig, use_container_width=True)
        return

    try:
        html = pio.to_html(
            fig,
            include_plotlyjs="inline",  # <- 핵심 (cdn 쓰면 하얀 화면 가능)
            full_html=False,
            auto_play=True,
        )
        components.html(html, height=height, scrolling=False)
    except Exception as e:
        st.error("Auto Play 렌더링 실패 → 일반 모드로 fallback 합니다.")
        st.exception(e)
        st.plotly_chart(fig, use_container_width=True)

def render_viewer_app():
    st.set_page_config(page_title="DRS Timeline Viewer", layout="wide")

    st.title("DRS Timeline Viewer")
    st.caption("FastF1 export CSV에서 `Time, DRS`로 DRS ON/OFF 타임라인을 재생합니다.")

    st.sidebar.header("CSV 업로드")
    uploaded = st.sidebar.file_uploader("FastF1 export CSV", type=["csv"])
    start = st.sidebar.button("Start", type="primary")
    auto_play = st.sidebar.checkbox("Auto Play", value=True)

    with st.expander("CSV 포맷 예시 보기 (FastF1 export 스타일)"):
        example = pd.DataFrame(
            {
                "Time": ["0 days 00:00:00", "0 days 00:00:00.123000", "0 days 00:00:00.304000"],
                "DRS": [1, 1, 0],
                "Speed": [0.0, 0.0, 0.0],
                "Throttle": [16.0, 16.0, 15.4],
                "Brake": [True, True, False],
                "X": [3435.50, 3435.43, 3436.00],
                "Y": [-2677.35, -2677.26, -2678.00],
            }
        )
        st.write("필수: `Time`, `DRS`")
        st.dataframe(example)

    if not start:
        st.info("CSV 업로드 후 Start를 누르세요.")
        return

    if uploaded is None:
        st.warning("CSV 파일을 업로드한 뒤 Start를 눌러주세요.")
        return

    df = load_drs_dataframe(uploaded)
    if df is None or df.empty:
        st.warning("유효한 DRS 데이터가 없습니다.")
        return

    anim_cfg = AnimConfig(max_frames=400)
    chart_cfg = DrsChartConfig(height=260)

    fig = build_drs_timeline_figure(
        df,
        anim_cfg=anim_cfg,
        chart_cfg=chart_cfg,
        show_static_bar=False,
    )
    
    if auto_play:
        html = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=False,
            auto_play=True,
        )
        components.html(html, height=chart_cfg.height + 80)
    else:
        auto_play = st.sidebar.checkbox("Auto Play", value=True)
        render_plotly(fig, height=chart_cfg.height + 80, auto_play=auto_play)


# =========================================================
# Entrypoint
# =========================================================
if __name__ == "__main__":
    render_viewer_app()