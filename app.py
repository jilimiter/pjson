import streamlit as st

st.title("🚗 GM SW Intern Streamlit Demo 5")

st.write("Hi! This is 5th App by using Streamlit.")

name = st.text_input("Input your name")

if name:
    st.success(f"Nice to meet you {name}! 🎉")

st.button("Button")