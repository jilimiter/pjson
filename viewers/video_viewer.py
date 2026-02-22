import streamlit as st

def render_reference_video():
    """
    유튜브 주행 영상을 대시보드에 배치하는 모듈입니다.
    """
    st.markdown("##### 📺 Reference Onboard (Suzuka)")
    
    # 2023/2025 Japan 데이터에 맞춘 수즈카 온보드 영상 ID입니다.
    video_url = "https://www.youtube.com/watch?v=5vTqY_nF7Sg"
    
    # 한 화면에 꽉 차게 보이도록 높이를 조절한 컨테이너입니다.
    with st.container():
        # 시작 지점을 데이터와 맞추고 싶다면 start_time 파라미터를 사용하세요.
        st.video(video_url)
        
    st.caption("Onboard Credit: Formula 1 Official")