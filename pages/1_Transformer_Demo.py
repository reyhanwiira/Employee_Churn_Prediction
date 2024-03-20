import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Transformer Demo", page_icon="🐍", layout="wide")

st.title("Employee Churn Prediction")
st.subheader("Predict Employee Churn using Transformer Pipeline")
st.write("This is a simple Machine Learning Web App to predict employee churn and retain your best employees")

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

def App1():
    # Load the pre-trained model
    with open('./pages/pipeline.pkl','rb') as f:
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
    
    col1, col2 = st.columns(2)
    with col1.container():
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

    with col2.container():
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
            show_prediction()

def App2():
    def process_data(data):
        # Load the model and perform predictions
        with open('pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        
        result = pipeline.predict(data)
        
        # Assign predictions based on result
        y_pred = ["Your employee may leave the company. (╥﹏╥)" if pred == 1 
                  else "Your employee may stay in the company. ⸜(｡ ˃ ᵕ ˂ )⸝♡" for pred in result]
        
        # Add predicted target to the data
        data['Predicted_target'] = y_pred
        return data
    
    # Streamlit app
    st.title("Predicting Employee Churn Using Machine Learning")
    sample =   {'Age':[28, 30, 35, 40, 45, 50, 18, 20, 22],
                'BusinessTravel':['Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Travel_Rarely', 'Travel_Rarely', 'Travel_Rarely', 'Non-Travel', 'Non-Travel', 'Non-Travel'],
                'Department':['HR', 'Corporate Functions', 'Sales', 'Delivery', 'Product', 'HR', 'Sales', 'Sales', 'Sales'],
                'EducationField':'Bachelors',
                'Gender':0,
                'MaritalStatus':'Single',
                'JobLevel':'L3',
                'OverTime':1,
                'JobInvolvement':3,
                'PerformanceRating':3,
                'MonthlyIncome':5000,
                'TotalWorkingYears':10,
                'TrainingTimesLastYear':2,
                'WorkLifeBalance':3,
                'YearsAtCompany':3,
                'YearsInCurrentRole':3,
                'YearsSinceLastPromotion':1,
                'YearsWithCurrManager':3,
                'JobSatisfaction':3,
                'RelationshipSatisfaction':3,
                'EnvironmentSatisfaction':3,
    }

    sample = pd.DataFrame(sample)
    st.dataframe(sample)

    # Create a button to go to link
    # Define the URL
    url = "https://www.kaggle.com/datasets/jash312/hr-employee-attrition-datasets?select=HR+Employee+data.csv"

    # Create a clickable link using markdown
    st.markdown(f"[Link to HR Employee Attrition Dataset]({url})")

    # Button to upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Load data from CSV
            data = pd.read_csv(uploaded_file)
            
            # Process the data
            processed_data = process_data(data)
            
            # Save the processed data to a CSV file
            st.write("Processed Data:")
            st.write(processed_data)
            st.write("Saving the processed data...")
            processed_data.to_csv('processed_data.csv', index=False)
            st.success("Data saved successfully! ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧")
        except Exception as e:
            st.error(f"Failed to open file: {e}")


def save_file(data):
    savepath = st.text_input("Enter file path to save (include.csv extension)")
    if st.button("Save File"):
        if savepath:
            try:
                data.to_csv(savepath, index=False)
                st.success("File Saved Successfully ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧")
            except Exception as e:
                st.error(f"Failed to save file:{e}")

c = st.container()
tab1, tab2 = st.tabs(["Single Prediction", "Predict Data using uploaded CSV"])
with tab1.container(): 
    App1()
with tab2.container():
    App2()

