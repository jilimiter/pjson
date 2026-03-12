import streamlit as st
import plotly.graph_objects as go

def render_bt_map(curr, data_pool):
    st.markdown('<div style="display: flex; flex-direction: column; flex: 1; height: 100%;">', unsafe_allow_html=True)
    idx = int(st.session_state.idx)
    trace = data_pool.iloc[max(0, idx-60):idx+1:2]

    # 제목과 현재 값을 같은 줄에 배치
    throttle_val = curr['throttle_t']
    brake_val = curr['brake_t']
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <h5 style="margin: 0;">🏁 Brake & Throttle</h5>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #888;">Current Input</div>
            <div style="font-size: 18px; font-weight: bold;"><span style="color: #55da55;">T: {throttle_val:.0f}%</span> / <span style="color: #d8412c;">B: {brake_val:.0f}%</span></div>
            <div style="font-size: 12px; color: #666;">&nbsp;</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trace['dist'], y=trace['throttle_t'], 
        name="THROTTLE",
        line=dict(color="#55da55", width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=trace['dist'], y=trace['brake_t'], 
        name="BRAKE", 
        fill='tozeroy', 
        fillcolor='rgba(255, 30, 0, 0.3)', 
        line=dict(color="#d8412c", width=2.5)
    ))

    fig.update_layout(
        height=150, 
        template="plotly_dark", 
        margin=dict(l=10, r=10, t=0, b=20),
        xaxis=dict(showgrid=False, title="Distance (m)", title_font=dict(size=10)),
        yaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#222", title_font=dict(size=10)),
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
    st.markdown('</div>', unsafe_allow_html=True)