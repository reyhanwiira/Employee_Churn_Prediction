import streamlit as st
import pandas as pd
import pickle
import time
import requests

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

st.set_page_config(page_title="Transformer - Single Prediction", layout="wide")

st.title("Predict your Employee Churn using Transformer Pipeline")
st.markdown("This is a simple Machine Learning Web App to predict employee Churn and retain your best employees")
# st.markdown("Employee turnover is divided into two categories – voluntary and involuntary. Employees who voluntarily resign fall are voluntary, while employees whose employment is terminated by the company is involuntary")
# st.markdown("**The implications of each category are divided this way:**")
# st.markdown("• Involuntary turnover indicates poor hiring. Reducing the involuntary turnover rate is a matter of hiring better candidates and to achieve that the recruitment team needs to improve its candidate screening processes.")
# st.markdown("• Voluntary turnover, on the other hand, indicates deeper issues within the organization and especially with the work culture.")
# multi = ('''    While both contribute to the overall turnover rate, companies must pay close attention to voluntary employee turnover because of its larger implications. There will always be some voluntary turnover, as employees will leave the company when they receive a better offer (in terms of pay, designation, role, and so on) and not necessarily because of issues with internal processes or management."
#     When the turnover rate is high and consistent, however, you know you have an internal problem. ''')
# st.markdown(multi)
st.write("Disclaimer: Don't rely too much on these prediction results, expert judgment is still needed to see any invisible influences")
df = pd.DataFrame({'Age',
                    'BusinessTravel',
                    'Department',
                    'EducationField',
                    'Gender',
                    'MaritalStatus',
                    'JobLevel',
                    'OverTime',
                    'JobInvolvement',
                    'PerformanceRating',
                    'TotalWorkingYears',
                    'TrainingTimesLastYear',
                    'WorkLifeBalance',
                    'YearsAtCompany',
                    'YearsInCurrentRole',
                    'YearsSinceLastPromotion',
                    'YearsWithCurrManager',
                    'JobSatisfaction',
                    'RelationshipSatisfaction',
                    'EnvironmentSatisfaction',
})
      
def save_file(data):
    savepath = st.text_input("Enter file path to save (include.csv extension)")
    if st.button("Save File"):
        if savepath:
            try:
                data.to_csv(savepath, index=False)
                st.success("File Saved Successfully ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧")
            except Exception as e:
                st.error(f"Failed to save file:{e}")

# Load the pre-trained model
with open('./ML_Models/pipeline.pkl','rb') as f:
    pipeline = pickle.load(f)

# Function to show prediction result
def show_prediction():
        p1 = float(e1)
        p2 = str(e2)
        p3 = str(e3)
        p4 = str(e4)
        p5 = str(e5)
        p6 = str(e6)
        p7 = str(e7)
        p8 = str(e8)
        p9 = float(e9)
        p10 = float(e10)
        p11 = float(e11)
        p12 = float(e12)
        p13 = float(e13)
        p14 = float(e14)
        p15 = float(e15)
        p16 = float(e16)
        p17 = float(e17)
        p18 = float(e18)
        p19 = float(e19)
        p20 = float(e20)
        p21 = float(e21)
    
        user_input = pd.DataFrame({'Age': [p1],
                            'BusinessTravel': [p2],
                            'Department': [p3],
                            'EducationField': [p4],
                            'Gender': [p5],
                            'MaritalStatus': [p6],
                            'JobLevel': [p7],
                            'OverTime': [p8],
                            'JobInvolvement': [p9],
                            'PerformanceRating': [p10],
                            'MonthlyIncome': [p11],
                            'TotalWorkingYears': [p12],
                            'TrainingTimesLastYear': [p13],
                            'WorkLifeBalance': [p14],
                            'YearsAtCompany': [p15],
                            'YearsInCurrentRole': [p16],
                            'YearsSinceLastPromotion': [p17],
                            'YearsWithCurrManager': [p18],
                            'JobSatisfaction': [p19],
                            'RelationshipSatisfaction': [p20], 
                            'EnvironmentSatisfaction': [p21],
                            })

        preview = pd.DataFrame(user_input)

        result = pipeline.predict(preview)[0]
        
        if result == 1:
            st.write("An employee may leave the organization.")
        else:
            st.write("An employee may stay with the organization.")

row1 = st.columns(1)
row2 = st.columns(1)

with row1[0]:
    st.subheader("Single Prediction")
    with st.container():
            e1 = st.slider("Age", 18, 60, 30)
            e9 = st.slider("Job Involvement", 1, 4, 2)
            e10 = st.slider("Performance Score", 1, 5, 3)
            e11 = st.slider("Monthly Income in $", 1000, 20000, 5000)
            e12 = st.slider("Total Working Years", 0, 40, 20)
            e13 = st.slider("Training Times Last Year", 0, 6, 2)
            e14 = st.slider("Work Life Balance", 1, 4, 2)
            e15 = st.slider("Years at Company", 1, 15, 7)
            e16 = st.slider("Years In Current Role", 1, 7, 2)
            e17 = st.slider("Years Since Last Promotion", 1, 6, 3)
            e18 = st.slider("Years With Current Manager", 1, 6, 3)
            e19 = st.slider("Job Satisfaction", 1, 4, 2)
            e20 = st.slider("Relationship Satisfaction", 1, 4, 2)
            e21 = st.slider("Environment Satisfaction", 1, 4, 2)

            options2 = ('Non-Travel', 'Travel_Rarely', 'Travel_Frequently')
            e2 = st.selectbox("Business Travel", options2)
    
            options3 = ('Corporate Functions', 'Delivery', 'HR', 'Marketing', 'Product', 'Sales')
            e3 = st.selectbox("Derpartment", options3)
            
            options4 = ('Diploma', 'Bachelors', 'Masters', 'Doctorate')
            e4 = st.selectbox("Education Field", options4)
            
            options5 = ('Female', 'Male')
            e5 = st.selectbox("Gender", options5)
            e5 = {'Female': 0, 'Male': 1}[e5]  # Label encoding
            
            options6 = ('Single', 'Married', 'Divorced')
            e6 = st.selectbox("Marital Status", options6)
            
            options7 = ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7')
            e7 = st.selectbox("Job Level", options7)
            
            options8 = ('No','Yes')
            e8 = st.selectbox("Over Time", options8)
            e8 = {'No': 0, 'Yes': 1}[e8]  # Label encoding  
with row2[0]:
        user_inputs = {
                        'Age': [e1],
                        'BusinessTravel': [e2],
                        'Department': [e3],
                        'EducationField': [e4],
                        'Gender': [e5],
                        'MaritalStatus': [e6],
                        'JobLevel': [e7],
                        'OverTime': [e8],
                        'JobInvolvement': [e9],
                        'PerformanceRating': [e10],
                        'MonthlyIncome': [e11],
                        'TotalWorkingYears': [e12],
                        'TrainingTimesLastYear': [e13],
                        'WorkLifeBalance': [e14],
                        'YearsAtCompany': [e15],
                        'YearsInCurrentRole': [e16],
                        'YearsSinceLastPromotion': [e17],
                        'YearsWithCurrManager': [e18],
                        'JobSatisfaction': [e19],
                        'RelationshipSatisfaction': [e20], 
                        'EnvironmentSatisfaction': [e21],
                    }

        sample = pd.DataFrame(user_inputs)

        st.write("Dataframe Preview")
        st.dataframe(sample)

        # Create a button to trigger prediction
        if st.button("Predict"):
            # progress_text = "Operation in progress. Please wait."
            # my_bar = st.progress(0, text=progress_text)

            # for percent_complete in range(100):
            #     time.sleep(0.01)
            #     my_bar.progress(percent_complete + 1, text=progress_text)
            # time.sleep(1)
            # my_bar.empty()
            #st.success('Done!')
            show_prediction()


