import streamlit as st
import plotly.graph_objects as go

def render_drs(curr, data_pool):
    """Render DRS status"""
    st.markdown("##### 🚀 DRS Status")
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    
    val = curr['drs_t']
    if val >= 10:
        status, color, shadow = "ON", "#00FF41", "#00FF41"
    elif val >= 8:
        status, color, shadow = "ENABLED", "#FFFF00", "#FFFF00"
    else:
        status, color, shadow = "OFF", "#FF1E00", "#FF1E00"
    
    st.markdown(f"""
        <style>
        .drs-button-wrapper {{
            background: linear-gradient(145deg, rgba(20,40,20,0.9) 0%, rgba(0,0,0,1) 100%);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 65, 0.2);
            height: 150px;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-around;
        }}
        .drs-btn {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 6px solid #222;
            background-color: #111;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 20px #000, 0 0 30px {shadow}66;
            border-color: {color};
            transition: all 0.3s;
        }}
        .btn-label {{ color: {color}; font-size: 10px; font-weight: bold; margin-bottom: -5px; }}
        .btn-status {{ font-family: 'DS-Digital', sans-serif; font-size: 32px; color: {color}; text-shadow: 0 0 15px {color}; }}
        .lap-badge {{
            background-color: rgba(0, 255, 65, 0.15);
            color: #00FF41;
            padding: 10px 30px;
            border-radius: 20px;
            font-family: 'DS-Digital', sans-serif;
            font-size: 32px;
            border: 1px solid rgba(0, 255, 65, 0.3);
        }}
        </style>
        <div class="drs-button-wrapper">
            <div class="drs-btn">
                <div class="btn-label">DRS</div>
                <div class="btn-status">{status}</div>
            </div>
            <div class="lap-badge">LAP {st.session_state.lap_count}</div>
        </div>
    """, unsafe_allow_html=True)
