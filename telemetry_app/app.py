# app.py
import streamlit as st
import pandas as pd

from utils.plotly_render import render_plotly
from utils.io_utils import read_csv_from_upload

from modules.DRS_viewer import (
    AnimConfig,
    DrsChartConfig,
    build_drs_dataframe_from_raw,
    build_drs_timeline_figure,
)

def ensure_state():
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "loaded" not in st.session_state:
        st.session_state.loaded = False
    if "load_err" not in st.session_state:
        st.session_state.load_err = None


def load_raw_df_once(uploaded):
    df, err = read_csv_from_upload(uploaded)

    if err:
        st.session_state.raw_df = None
        st.session_state.loaded = False
        st.session_state.load_err = err
        return

    st.session_state.raw_df = df
    st.session_state.loaded = True
    st.session_state.load_err = None


def main():
    st.set_page_config(page_title="DRS Timeline Viewer", layout="wide")
    ensure_state()

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

    if start:
        load_raw_df_once(uploaded)

    if not st.session_state.loaded:
        if st.session_state.load_err:
            st.warning(st.session_state.load_err)
        st.info("CSV 업로드 후 Start를 누르세요.")
        return

    raw_df = st.session_state.raw_df

    # viewer는 raw_df만 받음
    df, err = build_drs_dataframe_from_raw(raw_df)
    if err:
        st.error(err)
        return
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

    render_plotly(fig, height=chart_cfg.height + 80, auto_play=auto_play)


if __name__ == "__main__":
    main()