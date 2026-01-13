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
