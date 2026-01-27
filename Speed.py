# streamlit_Speed.py

import io
import pandas as pd
import streamlit as st

# =========================
# 1. 디지털 속도계 HTML 템플릿
# =========================
SPEED_TEMPLATE = """
<div style="
    width: 260px;
    height: 160px;
    border-radius: 130px 130px 0 0;
    background: radial-gradient(circle at 50% 0%, #555 0%, #000 60%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #00FF40;
    font-family: 'DS-Digital', 'Consolas', monospace;
    font-size: 86px;
    font-weight: bold;
    box-shadow: 0 6px 12px rgba(0,0,0,0.7);
    margin: 20px auto 10px auto;
">
  <span>{speed:>3d}</span>
  <span style="font-size:22px; margin-left:8px; align-self:flex-end; margin-bottom:24px;">km/h</span>
</div>
"""

# =========================
# 2. Streamlit 기본 설정
# =========================
st.set_page_config(page_title="Speed HUD Viewer", layout="centered")
st.title("Speed HUD – File Based")

st.sidebar.header("1. 데이터 파일 업로드")

uploaded = st.sidebar.file_uploader(
    "CSV / TSV 파일 업로드",
    type=["csv", "tsv", "txt"]
)

if uploaded is None:
    st.info("왼쪽에서 주행 데이터 파일(CSV/TSV)을 업로드해 주세요.")
    st.stop()

# =========================
# 3. 파일 읽기
# =========================
# 구분자 자동 추정 (쉼표/탭)
content = uploaded.read().decode("utf-8", errors="replace")
sep = "," if content.count(",") >= content.count("\t") else "\t"
df = pd.read_csv(io.StringIO(content), sep=sep)

st.sidebar.success(f"파일 로드 완료: {uploaded.name}")
st.sidebar.write(f"컬럼: {list(df.columns)}")

if df.empty:
    st.error("데이터가 비어 있습니다.")
    st.stop()

# =========================
# 4. 시간 / 속도 컬럼 선택
# =========================
st.sidebar.header("2. 컬럼 매핑")

time_col = st.sidebar.selectbox(
    "시간(Time) 컬럼 선택 (초 또는 ms)",
    options=df.columns,
    index=0
)

speed_col = st.sidebar.selectbox(
    "속도(Speed) 컬럼 선택 (km/h)",
    options=df.columns,
    index=min(1, len(df.columns)-1)
)

# 숫자형 변환 (에러는 NaN → 이후 dropna)
t = pd.to_numeric(df[time_col], errors="coerce")
v = pd.to_numeric(df[speed_col], errors="coerce")
mask = ~(t.isna() | v.isna())
t = t[mask].reset_index(drop=True)
v = v[mask].reset_index(drop=True)

if len(t) == 0:
    st.error("선택한 시간/속도 컬럼에 유효한 숫자 데이터가 없습니다.")
    st.stop()

# 시간 정규화 옵션 (0에서 시작)
st.sidebar.header("3. 시간 축 옵션")
normalize_time = st.sidebar.checkbox("시간을 0에서 시작하도록 정규화", value=True)

if normalize_time:
    t = t - t.iloc[0]

total_t = float(t.iloc[-1])

# =========================
# 5. 메인 UI – 슬라이더 + HUD
# =========================
st.subheader("Speed HUD")

time_value = st.slider(
    "Lap Time [s]",
    min_value=0.0,
    max_value=total_t,
    value=0.0,
    step=total_t / max(len(t), 1),
    format="%.3f",
)

# 슬라이더 시간에 가장 가까운 인덱스 찾기
idx = int((t - time_value).abs().argmin())

speed = int(v.iloc[idx])

# HUD 표시
st.markdown(SPEED_TEMPLATE.format(speed=speed), unsafe_allow_html=True)

# 부가 정보
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Time [s]", f"{float(t.iloc[idx]):.4f}")
with c2:
    st.metric("Speed [km/h]", f"{speed}")
with c3:
    st.metric("Index", f"{idx} / {len(t)-1}")

with st.expander("원본 데이터 미리보기"):
    st.dataframe(df.head(100))