import streamlit as st
import pandas as pd
import numpy as np
from viewers.score_viewer import render_similarity_score
from viewers.speed_viewer import render_speed_header
from viewers.brake_throttle_viewer import render_bt_map
from viewers.drs_viewer import render_drs
from viewers.video_viewer import render_reference_video

st.set_page_config(page_title="psjon telemetry", layout="wide")

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
    synced['score'] = (100 - np.abs(synced['speed_r']-synced['speed_t'])*1.5).clip(0,100).rolling(5, min_periods=1).mean()
    return synced

with st.sidebar:
    st.header("Let`s Win Max")
    ref_file = st.file_uploader("Ref", type=['csv'], label_visibility="collapsed")
    st.caption("Reference")
    tgt_file = st.file_uploader("Tgt", type=['csv'], label_visibility="collapsed")
    st.caption("Target")
    
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
    
    @st.fragment(run_every=0.3)
    def main_engine():
        if st.session_state.playing:
            new_idx = st.session_state.idx + 8
            if new_idx >= len(data_pool): 
                st.session_state.lap_count += 1
            st.session_state.idx = new_idx % len(data_pool)
        
        curr = data_pool.iloc[st.session_state.idx]
        render_similarity_score(curr)
        
        c1, c2, c3 = st.columns([1.1, 0.9, 1.2])
        with c1: render_speed_header(curr, data_pool)
        with c2: render_drs(curr, data_pool) 
        with c3: render_reference_video()
        
        st.markdown("---")
        render_bt_map(curr, data_pool)
        
    main_engine()