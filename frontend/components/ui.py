
import streamlit as st
def apply():
    with open("frontend/styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
