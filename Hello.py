import streamlit as st
from st_pages import Page, show_pages, add_page_title

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to My Demo App! 👋")

st.sidebar.success("Select an App you want to try.")

st.markdown(
    """
    This App was made for fullfilling the requirement to complete the course!

    **👈 Select an App on the left to get started.**
    
"""
)
