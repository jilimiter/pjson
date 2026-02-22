# utils/plotly_render.py
from __future__ import annotations

import json
from typing import Any, Optional, Tuple

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components


def extract_play_args(fig: go.Figure) -> Tuple[Optional[Any], dict]:
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


def render_plotly(fig: go.Figure, height: int, auto_play: bool, div_id: str = "drs_plotly_div") -> None:
    """
    Streamlit에서 Plotly figure를 렌더링.
    - auto_play=False: st.plotly_chart 사용
    - auto_play=True : HTML + JS로 Plotly.animate 호출
    """
    if not auto_play:
        st.plotly_chart(fig, use_container_width=True)
        return

    frame_sequence, options = extract_play_args(fig)

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

        if (document.readyState === "loading") {{
            document.addEventListener("DOMContentLoaded", run);
        }} else {{
            run();
        }}
    }})();
    </script>
    """

    components.html(html + autoplay_script, height=height, scrolling=False)
