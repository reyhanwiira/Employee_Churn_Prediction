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

    What can this application do?
    This app can predict employee attrition using Machine Learning (Transformer pipeline).

    **👈 Select an App on the left to get started.**

    
    
    There are many parameters affecting employee churn.

    *  Lack of career development opportunities: Employees often seek opportunities
    for career growth and development. When organizations fail to provide clear paths for advancement or invest in employee training and skill development, individuals may become dissatisfied and seek opportunities elsewhere. ()

    *  Inadequate compensation and benefits: Compensation and benefits play a significant role in employee satisfaction. If employees feel that their pay and benefits are not competitive or fair compared to industry standards, they may be more inclined to explore other opportunities that offer better financial rewards.

    *  Poor management and leadership: Ineffective or unsupportive management can contribute to high turnover. Employees may leave if they feel undervalued, unappreciated, or if there is a lack of clear communication and leadership within the organization. A positive and supportive managerial approach is crucial for retaining talent.

    *  Unhealthy work environment: A toxic or unhealthy work environment can drive employees away. Factors such as workplace harassment, discrimination, excessive stress, or a lack of work-life balance can negatively impact job satisfaction and contribute to a higher likelihood of turnover.

    *  Insufficient recognition and feedback: Employees who feel their contributions are not acknowledged or appreciated may become disengaged. Regular feedback, recognition programs, and a positive work culture that values and celebrates achievements are essential for employee retention.

    *  Limited flexibility and work-life balance: In today's workforce, flexibility and work-life balance are increasingly important. Organizations that do not offer flexible work arrangements or fail to support a healthy balance between work and personal life may experience higher turnover, especially among employees seeking more flexibility.

    *  Mismatched job expectations: If there is a significant disconnect between what employees expect from their roles and what the organization delivers, it can lead to dissatisfaction and increased turnover. Clear communication during the hiring process and ongoing dialogue about job expectations can help mitigate this factor.

""")

show_pages_from_config()