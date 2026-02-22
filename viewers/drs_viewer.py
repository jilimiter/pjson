import streamlit as st
import plotly.graph_objects as go

def render_drs(curr, data_pool):
    val = curr['drs_t']
    # 지수님 규칙: off 빨강 / enabled 노랑 / on 초록
    if val >= 10:
        status, color, shadow = "ON", "#00FF41", "#00FF41"
    elif val >= 8:
        status, color, shadow = "ENABLED", "#FFFF00", "#FFFF00"
    else:
        status, color, shadow = "OFF", "#FF1E00", "#FF1E00"
    
    # [디자인] 버튼 스타일의 UI와 LAP 정보 배치
    st.markdown(f"""
        <style>
        .drs-button-wrapper {{
            background: linear-gradient(145deg, rgba(20,40,20,0.9) 0%, rgba(0,0,0,1) 100%);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 65, 0.2);
            height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-around;
        }}
        .drs-btn {{
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 8px solid #222;
            background-color: #111;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 20px #000, 0 0 30px {shadow}66;
            border-color: {color};
            transition: all 0.3s;
        }}
        .btn-label {{ color: {color}; font-size: 12px; font-weight: bold; margin-bottom: -5px; }}
        .btn-status {{ font-family: 'DS-Digital', sans-serif; font-size: 45px; color: {color}; text-shadow: 0 0 15px {color}; }}
        .lap-badge {{
            background-color: rgba(0, 255, 65, 0.15);
            color: #00FF41;
            padding: 5px 25px;
            border-radius: 20px;
            font-family: 'DS-Digital', sans-serif;
            font-size: 28px;
            border: 1px solid rgba(0, 255, 65, 0.3);
            margin-top: 10px;
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

    # [개선] 트랙 맵 디자인 (배경 연두색 포인트 & 선 굵기 강화)
    fig_map = go.Figure()
    
    # 트랙 배경 반투명 연두색 글로우 효과
    fig_map.add_trace(go.Scatter(
        x=data_pool['x_t'], y=data_pool['y_t'],
        mode='lines', 
        line=dict(color='rgba(0, 255, 65, 0.1)', width=15), # 외곽 글로우
        hoverinfo='skip'
    ))

    # 메인 서킷 라인 (굵게)
    fig_map.add_trace(go.Scatter(
        x=data_pool['x_t'], y=data_pool['y_t'],
        mode='lines', 
        line=dict(color='#00FF41', width=5), # 선 굵기 5로 대폭 강화
        hoverinfo='skip'
    ))
    
    # Target 차량 위치 표시
    fig_map.add_trace(go.Scatter(
        x=[curr['x_t']], y=[curr['y_t']],
        mode='markers+text',
        text=["TARGET"],
        textposition="top center",
        textfont=dict(color="#321E8A", size=11, family="monospace"),
        marker=dict(size=15, color="#321E8A", symbol='circle', line=dict(width=2, color='white'))
    ))

    fig_map.update_layout(
        height=320,
        template="plotly_dark", 
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        showlegend=False, 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_map, use_container_width=True, config={'staticPlot': True})