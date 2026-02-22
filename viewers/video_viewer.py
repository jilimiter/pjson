import streamlit as st

def render_reference_video():
    """
    유튜브 주행 영상을 대시보드에 배치하는 모듈
    """
    st.markdown("##### 📺 Reference Onboard (Suzuka)")
    
    # 2023/2025 Japan 데이터에 맞춘 수즈카 온보드 영상 ID입니다.
    video_url = "https://www.youtube.com/watch?v=5vTqY_nF7Sg"
    
    with st.container():
        st.video(video_url)
        
    st.caption("Onboard Credit: Formula 1 Official")