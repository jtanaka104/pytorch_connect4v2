import streamlit as st

st.title("app.py の内容")

try:
    with open('app.py', encoding='utf-8') as f:
        code = f.read()
except FileNotFoundError:
    code = 'app.py が見つかりません。'

st.code(code, language='python')