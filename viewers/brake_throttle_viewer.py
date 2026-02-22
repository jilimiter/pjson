import streamlit as st
import plotly.graph_objects as go

def render_bt_map(curr, data_pool):
    idx = int(st.session_state.idx)
    trace = data_pool.iloc[max(0, idx-80):idx+1]

    st.markdown("##### 🏁 Brake & Throttle Overlay")
    
    fig = go.Figure()
    # Throttle: 녹색 선
    fig.add_trace(go.Scatter(
        x=trace['dist'], y=trace['throttle_t'], 
        name="THROTTLE", # 이름 추가
        line=dict(color="#55da55", width=2.5)
    ))
    
    # Brake: 빨간색 그라데이션
    fig.add_trace(go.Scatter(
        x=trace['dist'], y=trace['brake_t'], 
        name="BRAKE", # 이름 추가
        fill='tozeroy', 
        fillcolor='rgba(255, 30, 0, 0.3)', 
        line=dict(color="#d8412c", width=2.5)
    ))

    fig.update_layout(
        height=240, 
        template="plotly_dark", 
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, title="Distance (m)"),
        yaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#222"),
        showlegend=True, # 범례 표시 활성화
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})