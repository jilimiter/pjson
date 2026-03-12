import streamlit as st
import os

def render_reference_video():
    """
    유튜브 주행 영상을 대시보드에 배치하는 모듈
    """
    st.markdown("##### 📺 Reference Onboard")
    
    # 로컬 비디오 파일 경로
    video_path = "files/Max-Verstappen-s-Pole-Lap-2025-Japanese_sm.mp4"
    
    with st.container():
        if os.path.exists(video_path):
            # START 버튼이 눌렸을 때 자동재생
            autoplay = st.session_state.get('playing', False)
            st.video(video_path, autoplay=autoplay, loop=True)
        else:
            st.warning(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            st.info("파일이 pjson/files/ 폴더에 있는지 확인해주세요.")
    
    st.caption("Onboard Credit: Formula 1 Official")