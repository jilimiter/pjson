# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="F1 Telemetry Prototype",
    layout="wide",
)

# 필수 컬럼 (FastF1/기존 CSV 호환용)
REQUIRED_COLS = ["time_s", "brake_pct", "throttle_pct", "speed_kph", "drs"]
# 트랙용 선택 컬럼 (있으면 사용)
TRACK_COLS = ["x_m", "y_m"]

# -----------------------------------------
# CSV 로드
# -----------------------------------------
def load_from_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """
    CSV 형식의 로그 파일을 가정.
    필요한 표준 컬럼:
        time_s      : 시간 [s]
        brake_pct   : 브레이크 [%]
        throttle_pct: 스로틀 [%]
        speed_kph   : 속도 [km/h]
        drs         : DRS (0/1)

    선택 컬럼:
        x_m, y_m    : 트랙 상 차량 위치 [m]
    """
    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))

        # 실제 컬럼명을 표준 이름으로 매핑
        col_map = {}
        for col in df.columns:
            c = col.strip().lower()

            # 시간
            if c in ["time", "t", "time_s"]:
                col_map[col] = "time_s"

            # 제동
            elif "brake" in c:
                col_map[col] = "brake_pct"

            # 스로틀
            elif "throttle" in c:
                col_map[col] = "throttle_pct"

            # 속도
            elif "speed" in c:
                col_map[col] = "speed_kph"

            # DRS
            elif "drs" in c:
                col_map[col] = "drs"

            # 8자 트랙 X (여러 이름 허용)
            elif c in ["x", "x_m", "posx", "position_x", "world_x", "track_x"]:
                col_map[col] = "x_m"

            # 8자 트랙 Y (여러 이름 허용)
            elif c in ["y", "y_m", "posy", "position_y", "world_y", "track_y"]:
                col_map[col] = "y_m"

        df = df.rename(columns=col_map)

        # 필수 컬럼 체크
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(
                f"CSV에 필요한 컬럼이 없습니다: {missing}\n"
                f"필요 컬럼: {REQUIRED_COLS}"
            )
            return None

        # 타입 / 범위 정리
        df["time_s"] = df["time_s"].astype(float)
        df["speed_kph"] = df["speed_kph"].astype(float)
        df["brake_pct"] = df["brake_pct"].astype(float).clip(0, 100)
        df["throttle_pct"] = df["throttle_pct"].astype(float).clip(0, 100)
        df["drs"] = df["drs"].astype(int)

        # x_m, y_m 있으면 float로 캐스팅
        if "x_m" in df.columns:
            df["x_m"] = df["x_m"].astype(float)
        if "y_m" in df.columns:
            df["y_m"] = df["y_m"].astype(float)

        # time 기준 정렬
        df = df.sort_values("time_s").reset_index(drop=True)

        # 반환 컬럼: 필수 + (존재하는 트랙 컬럼)
        cols = REQUIRED_COLS.copy()
        for c in TRACK_COLS:
            if c in df.columns:
                cols.append(c)

        return df[cols]
    except Exception as e:
        st.error(f"파일 파싱 중 오류 발생: {e}")
        return None

def compute_similarity_score(df: pd.DataFrame) -> float:
    """지금은 더미 랜덤값."""
    return float(np.random.uniform(0.7, 0.95))

# -----------------------------------------
# 애니메이션용 다운샘플 공통 함수
# -----------------------------------------
def prepare_anim_data(df: pd.DataFrame, max_frames: int = 400):
    """
    Plotly 애니메이션 프레임 수를 제한하기 위해
    공통으로 사용하는 다운샘플/프레임 시간 계산.
    """
    n = len(df)
    step = max(1, n // max_frames)
    df_s = df.iloc[::step].reset_index(drop=True)

    t = df_s["time_s"].values
    if len(t) > 1:
        dt = float(np.mean(np.diff(t)))  # seconds
    else:
        dt = 0.05
    frame_duration_ms = max(10, int(dt * 1000))

    return df_s, frame_duration_ms

# -----------------------------------------
# Plotly 애니메이션 Figure (Speed + Brake/Throttle)
# -----------------------------------------
def make_speed_brake_anim(df: pd.DataFrame) -> go.Figure:
    """
    왼쪽 1/3 영역용 애니메이션:
    - Row 1: Speed [km/h]
    - Row 2: Brake[%] + Throttle[%]
    """
    df_s, frame_duration_ms = prepare_anim_data(df)
    t = df_s["time_s"].values
    speed = df_s["speed_kph"].values
    brake = df_s["brake_pct"].values
    throttle = df_s["throttle_pct"].values

    t0, t_end = float(t[0]), float(t[-1])

    # 전체 랩 기준 Speed y축 범위 계산
    speed_all = df["speed_kph"].values
    s_min = float(speed_all.min())
    s_max = float(speed_all.max())
    s_margin = max(5.0, 0.05 * (s_max - s_min))
    speed_min = s_min - s_margin
    speed_max = s_max + s_margin

    # Brake/Throttle는 0~100 고정
    bt_min, bt_max = 0, 100

    # 서브플롯 (2행, 공통 x축)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.05,
        subplot_titles=("Speed", "Brake / Throttle"),
    )

    # 초기 프레임 (첫 포인트만)
    sub_idx = 1
    sub_t = t[:sub_idx]
    fig.add_trace(
        go.Scatter(
            x=sub_t,
            y=speed[:sub_idx],
            name="Speed [km/h]",
            line=dict(color="royalblue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sub_t,
            y=brake[:sub_idx],
            name="Brake [%]",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sub_t,
            y=throttle[:sub_idx],
            name="Throttle [%]",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    # 프레임들 정의 (지금까지의 시계열)
    frames = []
    for i in range(1, len(df_s)):
        sub_t = t[: i + 1]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=sub_t, y=speed[: i + 1]),
                    go.Scatter(x=sub_t, y=brake[: i + 1]),
                    go.Scatter(x=sub_t, y=throttle[: i + 1]),
                ],
                name=str(i),
            )
        )

    fig.frames = frames

    # 축 범위 고정 (전체 랩 기준 “이빠이”)
    fig.update_yaxes(
        title_text="Speed [km/h]", range=[speed_min, speed_max], row=1, col=1
    )
    fig.update_yaxes(title_text="[%]", range=[bt_min, bt_max], row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", range=[t0, t_end], row=2, col=1)
    fig.update_xaxes(range=[t0, t_end], row=1, col=1)

    # 애니메이션 버튼 & 슬라이더
    fig.update_layout(
        height=450,
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
                        "args":[
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{t[i]:.2f}s",
                    }
                    for i in range(len(df_s))
                ],
                "x": 0.0,
                "y": -0.10,
                "len": 1.0,
            }
        ],
    )

    # 자동 재생 설정
    fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True

    return fig

# -----------------------------------------
# 8자 트랙 + 현재 위치 애니메이션 Figure
# -----------------------------------------
def make_track_anim(df: pd.DataFrame) -> go.Figure:
    """
    x_m, y_m가 있을 때만 사용.
    - 회색 선: 전체 8자 트랙
    - 초록 점: 현재 위치 (애니메이션)
    """
    if "x_m" not in df.columns or "y_m" not in df.columns:
        raise ValueError("x_m / y_m 컬럼이 없습니다.")

    df_s, frame_duration_ms = prepare_anim_data(df)

    x_all = df["x_m"].values
    y_all = df["y_m"].values
    x_s = df_s["x_m"].values
    y_s = df_s["y_m"].values
    t_s = df_s["time_s"].values

    # 축 범위 (여유 5%)
    xmin, xmax = float(x_all.min()), float(x_all.max())
    ymin, ymax = float(y_all.min()), float(y_all.max())
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0
    xm = 0.05 * dx
    ym = 0.05 * dy

    fig = go.Figure()

    # 전체 트랙 (고정)
    fig.add_trace(
        go.Scatter(
            x=x_all,
            y=y_all,
            mode="lines",
            name="Track",
            line=dict(color="lightgray", width=2),
        )
    )

    # 현재 위치 점 (초기 한 점)
    fig.add_trace(
        go.Scatter(
            x=[x_s[0]],
            y=[y_s[0]],
            mode="markers",
            name="Car",
            marker=dict(color="limegreen", size=10),
        )
    )

    # 프레임들 정의 (점만 이동)
    frames = []
    for i in range(len(df_s)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=x_all, y=y_all),  # 트랙 (고정)
                    go.Scatter(x=[x_s[i]], y=[y_s[i]]),  # 현재 위치
                ],
                name=str(i),
            )
        )

    fig.frames = frames

    fig.update_layout(
        title="Track View (Figure-8 with Current Position)",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        height=350,
        margin=dict(t=40, b=40, l=40, r=20),
        showlegend=False,
        xaxis=dict(range=[xmin - xm, xmax + xm]),
        yaxis=dict(range=[ymin - ym, ymax + ym], scaleanchor="x", scaleratio=1),
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
                    for i in range(len(df_s))
                ],
                "x": 0.0,
                "y": -0.10,
                "len": 1.0,
            }
        ],
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True

    return fig

# -----------------------------------------
# DRS 타임라인(정적) Figure
# -----------------------------------------
def make_drs_timeline(df: pd.DataFrame) -> go.Figure:
    t = df["time_s"].values
    drs = df["drs"].values

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=t,
            y=drs,
            marker_color="limegreen",
            name="DRS",
        )
    )
    fig.update_layout(
        title="DRS ON/OFF Timeline",
        xaxis_title="Time [s]",
        yaxis_title="DRS (0=OFF, 1=ON)",
        yaxis=dict(range=[-0.1, 1.1]),
        height=250,
        margin=dict(t=40, b=40, l=40, r=20),
        showlegend=False,
    )
    return fig

# =========================================
# 메인 UI
# =========================================
st.title("F1 Telemetry Viewer (CSV + Animation Layout)")
st.caption(
    "왼쪽: Speed + Brake/Throttle 애니메이션 (1배속)\n"
    "가운데: DRS ON/OFF 타임라인\n"
    "오른쪽(우상단): 8자 트랙 뷰 + 현재 위치, 아래 요약 정보"
)

st.sidebar.header("1. 로그 파일 선택 (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "예: dummy_telemetry.csv 또는 FastF1에서 export한 CSV", type=["csv"]
)
start_clicked = st.sidebar.button("Start (Animate)", type="primary")

with st.expander("CSV 포맷 예시 보기"):
    example = pd.DataFrame(
        {
            "time_s": [0.0, 0.05, 0.10],
            "brake_pct": [0.0, 3.2, 5.0],
            "throttle_pct": [80.0, 82.1, 85.3],
            "speed_kph": [120.0, 123.4, 127.8],
            "drs": [0, 0, 1],
            "x_m": [0.0, 10.0, 20.0],
            "y_m": [0.0, 5.0, 0.0],
        }
    )
    st.write(example)

if start_clicked:
    if uploaded_file is None:
        st.warning("CSV 파일을 업로드한 뒤 Start를 눌러주세요.")
        st.stop()

    df = load_from_uploaded_file(uploaded_file)
    if df is None:
        st.stop()

    # Figure 생성
    fig_anim = make_speed_brake_anim(df)
    fig_drs = make_drs_timeline(df)
    similarity = compute_similarity_score(df)

    has_track = "x_m" in df.columns and "y_m" in df.columns
    fig_track = make_track_anim(df) if has_track else None

    # ---- 레이아웃: 왼쪽, 가운데, 오른쪽(우상단 트랙 + 요약) ----
    col_left, col_mid, col_right = st.columns([1.2, 1.0, 0.9])

    with col_left:
        st.subheader("Playback (Speed / Brake / Throttle)")
        st.plotly_chart(fig_anim, use_container_width=True)

    with col_mid:
        st.subheader("DRS ON/OFF")
        st.plotly_chart(fig_drs, use_container_width=True)

    with col_right:
        st.subheader("Track (Figure-8 with Car Position)")
        if has_track:
            st.plotly_chart(fig_track, use_container_width=True)
        else:
            st.info("이 CSV에는 x_m / y_m 트랙 좌표가 없어 트랙 뷰를 표시하지 않습니다.")

        st.subheader("Summary")
        st.markdown(f"**Similarity Score** (dummy): `{similarity:.2f}`")
        st.markdown("---")
        st.write(
            "- 나중에 여기에는 랩 타임, 섹터 타임,\n"
            "  코너별 손실 순위 같은 요약 정보 넣으면 좋음."
        )

else:
    st.info(
        "왼쪽에서 CSV를 업로드하고 **Start (Animate)** 버튼을 누르면\n"
        "Speed/Brake/Throttle 애니메이션, 8자 트랙 뷰, DRS 타임라인이 표시됩니다.\n\n"
        "`generate_dummy_csv.py`로 만든 `dummy_telemetry.csv`를 테스트용으로 사용할 수 있습니다."
    )
