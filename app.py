import streamlit as st
import pandas as pd
import numpy as np
from viewers.score_viewer import render_similarity_score
from viewers.speed_viewer import render_speed_header
from viewers.brake_throttle_viewer import render_bt_map
from viewers.drs_viewer import render_drs
from viewers.video_viewer import render_reference_video
from viewers.track_map_viewer import render_track_map

st.set_page_config(page_title="psjon telemetry", layout="wide")

# 스크롤 제거를 위한 CSS
st.markdown("""
<style>
    /* 상단 메뉴바 숨기기 */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .stAppHeader {
        display: none !important;
    }
    
    .st-emotion-cache-10p9htt {
        display: none !important;
    }
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    .stMainBlockContainer {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    h5 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    /* 각 영역 높이 통일 */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="stVerticalBlock"] {
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    
    /* Streamlit 푸터 숨기기 */
    footer {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_synced_data(ref, tgt):
    df_r = pd.read_csv(ref)
    df_t = pd.read_csv(tgt)
    max_d = min(df_r['Distance'].max(), df_t['Distance'].max())
    grid = np.arange(0, max_d, 5) 

    def get_safe_interp(target_df, col_names, default=0):
        col = next((c for c in col_names if c in target_df.columns), None)
        if col is not None:
            vals = target_df[col]
            if vals.dtype == bool: vals = vals.astype(float) * 100.0
            elif vals.max() <= 1.5 and col.lower() in ['brake', 'throttle']: vals *= 100.0
            return np.interp(grid, target_df['Distance'], vals)
        return np.full(len(grid), float(default))

    synced = pd.DataFrame({
        'dist': grid,
        'speed_r': np.interp(grid, df_r['Distance'], df_r['Speed']),
        'speed_t': np.interp(grid, df_t['Distance'], df_t['Speed']),
        'drs_t': get_safe_interp(df_t, ['DRS', 'drs']),
        'brake_t': get_safe_interp(df_t, ['Brake', 'brake']),
        'throttle_t': get_safe_interp(df_t, ['Throttle', 'throttle']),
        'x_t': get_safe_interp(df_t, ['X', 'x']), 
        'y_t': get_safe_interp(df_t, ['Y', 'y'])
    })
    synced['score'] = (100 - np.abs(synced['speed_r']-synced['speed_t'])*1.5).rolling(5, min_periods=1).mean()
    return synced

with st.sidebar:
    st.header("Let's Win Max")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Reference")
    ref_file = st.file_uploader("Ref", type=['csv'], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Target")
    tgt_file = st.file_uploader("Tgt", type=['csv'], label_visibility="collapsed")
    
    if "playing" not in st.session_state: st.session_state.playing = False
    if "idx" not in st.session_state: st.session_state.idx = 0
    if "lap_count" not in st.session_state: st.session_state.lap_count = 1 # Lap 초기화
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("🚀 START", use_container_width=True): st.session_state.playing = True
    if c2.button("🛑 STOP", use_container_width=True): st.session_state.playing = False
    if st.button("🔄 RESET ALL", use_container_width=True):
        st.session_state.idx = 0
        st.session_state.lap_count = 1
        st.session_state.playing = False
        st.rerun()

if ref_file and tgt_file:
    data_pool = get_synced_data(ref_file, tgt_file)
    
    @st.fragment(run_every=0.5)
    def main_engine():
        if st.session_state.playing:
            new_idx = st.session_state.idx + 8
            if new_idx >= len(data_pool): 
                st.session_state.lap_count += 1
            st.session_state.idx = new_idx % len(data_pool)
        
        curr = data_pool.iloc[st.session_state.idx]
        
        # 상단 영역: MATCH PERFORMANCE + DRS
        top_col1, top_col2 = st.columns([2, 1])
        with top_col1: render_similarity_score(curr)
        with top_col2: render_drs(curr, data_pool)
        
        # 중간 영역: Reference Video + Track Map + Empty
        middle_col1, middle_col2 = st.columns([2, 1])
        with middle_col1: 
            with st.container():
                render_track_map(curr, data_pool)
        with middle_col2:
            with st.container():
                render_reference_video()
        
        # 하단 영역: Speed Analysis + Brake & Throttle
        bottom_col1, bottom_col2 = st.columns([1, 1])
        with bottom_col1: render_speed_header(curr, data_pool)
        with bottom_col2: render_bt_map(curr, data_pool)
        
    main_engine()