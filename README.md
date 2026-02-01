# Live Brake / Throttle Telemetry Viewer

브레이크와 스로틀 텔레메트리 데이터를 시각화하고  
여러 개의 주행 데이터를 **겹쳐서(overlay)** 또는 **나란히(side-by-side)** 비교할 수 있는  
Streamlit 기반 분석 도구입니다.

자동차 주행 데이터, 시뮬레이션 결과, 로깅 데이터 분석을 목적으로 제작되었습니다.

---

앱 실행 시
좌측 Bar에서 Telemetry Input 파일을 업로드하고
아래 설정 완료 후 Build Comparison하면 됩니다.
Compare mode에 Show side-by-side view 선택하면 업로드한 파일을 따로따로 볼 수 있어 편리합니다.
Build Comparison 누르기 전에 우측 메인 화면에 해당 파일 미리보기가 뜹니다.

Build되면,
Overlay comparison (default)에서는 업로드한 모든 데이터가 합쳐져 보입니다.
Side-by-side view에서는 각가의 데이터를 따로따로 볼 수 있습니다.
비교 그래프 좌측 상단에 Play/Pause 버튼을 눌러 데이터를 확인하거나, 상단 바를 직접 움직여 원하는 위치에서의 차량 데이터를 확인할 수 있습니다.
우측 상단에 도구를 활용하여 그래프 캡쳐, 확대/축소, 전체보기 등 기능을 이용하실 수 있으며,
코스 맵/그래프 화면은 드래그하여 확대, 더블클릭하여 초기화면대로 축소가 가능합니다.


---

## 실행 방법 (Windows / PowerShell 기준)

### 1. 가상환경 생성
```powershell
python -m venv .venv
.\.venv\Scripts\activate


2. 라이브러리 설치

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


3. 앱 실행
streamlit run liveBrakeThrottleMap.py

---

## 주요 기능

### 1. 다중 데이터 비교
- CSV / XLSX 형식의 텔레메트리 파일 **2개 이상 업로드 가능**
- 기본 비교 방식: **Overlay 비교**
- 버튼을 통해 **Side-by-Side 비교 모드**로 전환 가능

### 2. 2D 트랙 시각화
- `X`, `Y` 좌표를 이용한 **2D 트랙 맵**
- 주행 진행에 따라 차량 위치가 실시간으로 이동
- 여러 데이터의 주행 궤적을 동시에 비교 가능

### 3. 브레이크 / 스로틀 분석
- Brake / Throttle 값을 시간 또는 거리 기준으로 표시
- **Rolling window 방식**으로 현재 구간을 고해상도로 확대 표시
- Plotly 기반 인터랙션 제공
  - 줌 / 팬
  - Reset axes (축 원래 상태로 복원)

### 4. 비교 기준 선택
- 비교 x축 기준 선택 가능
  - Distance (권장)
  - Time
  - SessionTime

---

## 입력 데이터 형식

업로드하는 파일은 아래 컬럼을 포함해야 합니다.

### 필수 컬럼
- `Time`
- `Throttle`
- `Brake`
- `X`
- `Y`

### 권장 컬럼
- `Distance`
- `SessionTime`
- `LapNumber`
- `Status`

※ `Distance` 컬럼을 사용할 경우 가장 직관적인 비교가 가능합니다.

----------------------------

# Streamlit Project

인턴 교육용으로 만든 가장 간단한 Streamlit 예제 프로젝트입니다.  
clone → venv → install → run 과정을 경험하는 것이 목표입니다.

---

## 프로젝트 구조

```
sw-internship/
├── app.py
├── requirements.txt
└── README.md
```

---

## 실행 방법

### 1. 저장소 클론
```bash
git clone <레포주소>
cd sw-internship
```

---

### 2. 가상환경(venv) 생성
```bash
python -m venv venv
```

---

### 3. 가상환경 활성화

#### Windows
```bash
venv\Scripts\activate
```

성공하면 터미널에 `(venv)` 표시가 나타납니다.

---

### 4. 라이브러리 설치
```bash
pip install -r requirements.txt
```

---

### 5. 실행
```bash
streamlit run app.py
```

브라우저가 자동으로 열리면서 다음과 같은 화면이 나타납니다.
- 제목
- 텍스트 입력창
- 버튼

---

## 문제가 생겼을 때

### python 명령어가 안 될 때
```bash
python --version
```
또는
```bash
py --version
```

### pip이 안 될 때
```bash
python -m pip install -r requirements.txt
```

### Streamlit 실행이 안 될 때
```bash
pip show streamlit
```
