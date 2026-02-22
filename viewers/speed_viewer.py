import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_speed_header(curr, data_pool):
    diff = curr['speed_t'] - curr['speed_r']
    st.markdown(f"##### 🚀 Speed Analysis")
    
    col_g, col_m = st.columns([1.2, 0.8])
    with col_g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=curr['speed_t'],
            number={'suffix': " km/h", 'font': {'size': 22}},
            gauge={'axis': {'range': [0, 360]}, 'bar': {'color': "#FF1E00"}}))
        fig.update_layout(height=160, margin=dict(l=10,r=10,t=10,b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col_m:
        st.metric("Target", f"{curr['speed_t']:.1f}", delta=f"{diff:+.1f}")
        st.caption(f"Ref: {curr['speed_r']:.1f} km/h")

    idx = int(st.session_state.idx)
    trace = data_pool.iloc[max(0, idx-60):idx+1][['speed_r', 'speed_t']].copy().reset_index(drop=True)

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
        height=180, 
        template="plotly_dark", 
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#333", autorange=True),
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    )
    st.plotly_chart(fig_trace, use_container_width=True, config={'staticPlot': True})