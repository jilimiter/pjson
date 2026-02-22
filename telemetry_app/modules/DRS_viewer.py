# modules/DRS_viewer.py
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
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
def build_drs_dataframe_from_raw(raw_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    app에서 읽어둔 raw_df를 받아서 DRS 전용 df로 변환.
    반환:
      - (df, None) 성공
      - (None, err) 실패
    """
    try:
        if raw_df is None or raw_df.empty:
            return None, "raw_df가 비어있습니다."

        err = validate_required_columns(raw_df, REQUIRED_COLS_DRS)
        if err:
            return None, err

        df = parse_time_to_timedelta(raw_df, "Time")
        if df["Time"].isna().all():
            return None, "Time 컬럼을 timedelta로 파싱할 수 없습니다."

        df = df.sort_values("Time").reset_index(drop=True)

        drs01 = normalize_drs_to_01(df["DRS"])

        out = pd.DataFrame(
            {
                "Time": df["Time"],
                "time_s": df["Time"].dt.total_seconds().astype(float),
                "DRS": drs01.astype(int),
            }
        )
        return out, None

    except Exception as e:
        return None, f"파일 파싱 중 오류 발생: {e}"


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

    if s.dtype == bool or str(s.dtype).lower() == "boolean":
        arr = s.astype(int).to_numpy()

    elif s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        if ss.isin(["true", "false"]).all():
            arr = (ss == "true").astype(int).to_numpy()
        else:
            arr = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()

    else:
        arr = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()

    return np.clip(arr, 0, 1)

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
    원본 build_drs_timeline_figure 그대로.
    """
    t_full = df["time_s"].to_numpy()
    drs_full = df["DRS"].to_numpy()

    df_s, frame_duration_ms = downsample_for_animation(df, anim_cfg)
    t_s = df_s["time_s"].to_numpy()

    t0, t_end = float(t_full[0]), float(t_full[-1])
    y0, y1 = chart_cfg.y_min, chart_cfg.y_max

    fig = go.Figure()

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

    cursor_x = float(t_s[0]) if len(t_s) else t0
    cursor_y = float(df_s["DRS"].iloc[0]) if len(df_s) else 0

    fig.add_trace(
        go.Scatter(
            x=[cursor_x],
            y=[cursor_y],
            mode="markers",
            name="Current",
            showlegend=False,
            marker=dict(size=14, opacity=1.0, color="red"),
        )
    )

    frames = []
    for i in range(len(t_s)):
        cx = float(t_s[i])
        cy = float(df_s["DRS"].iloc[i])
        frames.append(
            go.Frame(
                data=[go.Scatter(x=[cx], y=[cy])],
                traces=[1],
                name=str(i),
            )
        )
    fig.frames = frames

    fig.update_layout(
        title="DRS ON/OFF Timeline",
        xaxis_title="Time [s]",
        yaxis_title="DRS (0=OFF, 1=ON)",
        yaxis=dict(
            range=[y0, y1],
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["OFF", "ON"],
        ),
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