🚀 F1 Telemetry Dashboard

실시간 F1 텔레메트리 데이터를 기반으로
속도, 랩 타임, RPM, 기어, 그리고 2D 트랙 위치를 시각화하는 Streamlit 대시보드입니다.

📌 주요 기능

🏎️ 실시간 Speed HUD (게이지 UI)

⏱️ Lap 기반 누적 시간 (Cumulative Time) 계산

📈 속도 그래프 + 현재 시점 빨간 점 표시

🗺️ 2D 트랙 지도 위 차량 위치 실시간 표시

▶️ Start / Stop / Reset 자동 재생 기능

⚡ st.fragment 기반 고속 자동 업데이트

🧠 데이터 처리 방식

Time + LapNumber를 이용해 연속 시간축(cum_sec) 생성

X/Y 좌표는 보간(interpolation)하여 끊김 최소화

좌표 결측 시 마지막 유효 위치 유지
