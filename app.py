import streamlit as st

st.set_page_config( page_title = "To-Do", page_icon = "📝", layout = "centered")

default_todo_list = ["모닝 Toilet Routine 수행", "출근하기", "퇴근하기", "저녁밥 먹기"]

# 초기화
if "todos" not in st.session_state:
    st.session_state.todos = default_todo_list.copy()

if "new_todo" not in st.session_state:
    st.session_state.new_todo = ""


def add_todo():
    text = st.session_state.new_todo.strip()
    if text:
        st.session_state.todos.append(text)
        st.session_state.new_todo = ""


st.title("📝 To‑do")

st.text_input("새 To‑do 입력", key="new_todo")
st.button("Add", on_click=add_todo)

st.divider()


for todo in st.session_state.todos:
    st.write(f"- {todo}")
