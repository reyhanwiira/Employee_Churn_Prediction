import streamlit as st
from st_pages import show_pages_from_config

# st.set_page_config(
#     page_title="Home",
#     page_icon="👋",
# )

st.write("# Welcome to My Demo App 🤗")

st.sidebar.success("Select an App you want to try.")
#st.sidebar.info("Transformers Demo, is an app that predict employee attrition using Transformer pipeline.")
st.markdown(
    """
    This App was made for fullfilling the requirement to complete the course!

    **👈 Select an App on the left to get started.**
""")

show_pages_from_config()