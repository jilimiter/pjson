import streamlit as st
import plotly.graph_objects as go

def render_track_map(curr, data_pool):
    """Render track map with current position"""
    st.markdown('<div style="display: flex; flex-direction: column; flex: 1; height: 100%;">', unsafe_allow_html=True)
    st.markdown("##### 🗺️ Track Position")
    
    # 전체 트랙 그리기
    fig = go.Figure()
    
    # 전체 트랙 (데이터 포인트 줄이기)
    track_data = data_pool[::3]  # 3개 중 1개만 사용
    fig.add_trace(go.Scatter(
        x=track_data['x_t'],
        y=track_data['y_t'],
        mode='lines',
        line=dict(color='#00ff00', width=3),
        name='Track',
        showlegend=False
    ))
    
    # 현재 위치
    fig.add_trace(go.Scatter(
        x=[curr['x_t']],
        y=[curr['y_t']],
        mode='markers+text',
        marker=dict(size=15, color='#ff0000', symbol='star'),
        text=['TARGET'],
        textposition='top center',
        textfont=dict(color='#ff0000', size=10),
        name='Current',
        showlegend=False
    ))
    
    fig.update_layout(
        height=220,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True}, key='track_map_chart')
    st.markdown('</div>', unsafe_allow_html=True)
