# app.py
import io
import json
import streamlit as st
import pandas as pd
import plotly.io as pio
import streamlit.components.v1 as components
import plotly.graph_objects as go

from DRS_viewer import (
    AnimConfig,
    DrsChartConfig,
    build_drs_dataframe_from_raw,
    build_drs_timeline_figure,
)

def _extract_play_args(fig: go.Figure):
    """
    fig.layout.updatemenus[0].buttons[0] (Play 버튼)의 animate args를 그대로 뽑아온다.
    실패하면 안전한 기본값으로 fallback.
    """
    try:
        updatemenus = fig.layout.updatemenus
        if not updatemenus:
            raise KeyError("No updatemenus")

        buttons = updatemenus[0].buttons
        if not buttons:
            raise KeyError("No buttons")

        play_btn = buttons[0]
        # args 구조: [None, { ...options... }]
        args = play_btn.args
        if not isinstance(args, (list, tuple)) or len(args) < 2:
            raise ValueError("Unexpected args format")

        frame_sequence = args[0]  
        options = args[1]         
        return frame_sequence, options

    except Exception:
        return None, {
            "frame": {"duration": 16, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 0},
            "mode": "immediate",
        }


def render_plotly(fig: go.Figure, height: int, auto_play: bool):
    if not auto_play:
        st.plotly_chart(fig, use_container_width=True)
        return

    frame_sequence, options = _extract_play_args(fig)

    div_id = "drs_plotly_div"

    html = pio.to_html(
        fig,
        include_plotlyjs="inline",
        full_html=False,
        auto_play=False,
        div_id=div_id,
    )

    js_frame_seq = json.dumps(frame_sequence)
    js_options = json.dumps(options)

    autoplay_script = f"""
    <script>
    (function() {{
        const divId = "{div_id}";
        const frameSeq = {js_frame_seq};
        const options = {js_options};

        function run() {{
            const gd = document.getElementById(divId);
            if (!gd) return;

            // plotly가 fully ready 된 뒤 animate 호출
            const start = () => {{
                try {{
                    Plotly.animate(gd, frameSeq, options);
                }} catch (e) {{
                    console.error("Autoplay animate failed:", e);
                }}
            }};

            if (gd.data && gd.layout) {{
                start();
            }} else {{
                gd.on('plotly_afterplot', start);
            }}
        }}

        // Streamlit components가 DOM을 붙인 직후 실행
        if (document.readyState === "loading") {{
            document.addEventListener("DOMContentLoaded", run);
        }} else {{
            run();
        }}
    }})();
    </script>
    """

    components.html(html + autoplay_script, height=height, scrolling=False)


def ensure_state():
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "loaded" not in st.session_state:
        st.session_state.loaded = False
    if "load_err" not in st.session_state:
        st.session_state.load_err = None


def load_raw_df_once(uploaded):
    """
    Start 누를 때만 호출.
    uploaded_file -> raw_df를 딱 1번 읽어서 session_state에 저장
    """
    if uploaded is None:
        st.session_state.raw_df = None
        st.session_state.loaded = False
        st.session_state.load_err = "CSV 파일을 업로드한 뒤 Start를 눌러주세요."
        return

    try:
        content = uploaded.read()  # 데이터 읽음
        raw_df = pd.read_csv(io.BytesIO(content))

        st.session_state.raw_df = raw_df
        st.session_state.loaded = True
        st.session_state.load_err = None

    except Exception as e:
        st.session_state.raw_df = None
        st.session_state.loaded = False
        st.session_state.load_err = f"CSV 읽기 실패: {e}"


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