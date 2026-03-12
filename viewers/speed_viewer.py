import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_speed_header(curr, data_pool):
    """Render speed comparison header"""
    st.markdown('<div style="display: flex; flex-direction: column; flex: 1; height: 100%;">', unsafe_allow_html=True)
    
    # 제목과 속도 정보를 같은 줄에 배치
    speed_r = curr['speed_r']
    speed_t = curr['speed_t']
    diff = speed_t - speed_r
    color = "#00ff00" if diff >= 0 else "#ff0000"
    arrow = "↑" if diff >= 0 else "↓"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <h5 style="margin: 0;">🏎️ Speed Analysis</h5>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #888;">Target Speed</div>
            <div style="font-size: 18px; font-weight: bold;">{speed_t:.1f} km/h</div>
            <div style="font-size: 12px; color: {color};">{arrow} {abs(diff):.1f} vs Ref</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    idx = int(st.session_state.idx)
    trace = data_pool.iloc[max(0, idx-40):idx+1:2][['speed_r', 'speed_t']].copy().reset_index(drop=True)

    fig_trace = go.Figure()
    fig_trace.add_trace(go.Scatter(
        y=trace['speed_r'], 
        name='REF', 
        fill='tozeroy', 
        line=dict(width=0), 
        fillcolor='rgba(135, 206, 250, 0.4)'
    ))
    fig_trace.add_trace(go.Scatter(
        y=trace['speed_t'], 
        name='TARGET', 
        line=dict(color='#0088ff', width=3, shape='spline')
    ))

    fig_trace.update_layout(
        height=150, 
        template="plotly_dark", 
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#333", autorange=True),
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
    )
    st.plotly_chart(fig_trace, use_container_width=True, config={'staticPlot': True})
    st.markdown('</div>', unsafe_allow_html=True)