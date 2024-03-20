from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch
import torch.nn as nn
import time

# class DNN(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2):
#         super(DNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, 1)
#         self.activation = nn.ReLU()  # Changed activation to ReLU for hidden layers
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out = self.activation(self.fc1(x))
#         out = self.activation(self.fc2(out))
#         out = self.sigmoid(self.fc3(out))
#         return out

# Load Our pre-trained model
PATH = "./pages/churn-model-full.pt"
model = torch.load(PATH)
model.eval()

sample = {
         'Age': [28, 30, 21, 25, 40, 42],
         'BusinessTravel': [0,0,1,1,0,0],
         'Department': [0,3,1,3,0,0],
         'EducationField': [0,3,1,3,0,0],
         'Gender': [0,1,0,1,1,0],
         'MaritalStatus': [0,3,1,3,0,0],
         'JobLevel': [0,3,1,3,0,0],
         'OverTime': [0,1,0,1,1,0],
         'JobInvolvement': [2,2,2,2,2,2],
         'PerformanceRating': [3,3,3,3,3,3],
         'MonthlyIncome': [2000, 2000, 2000, 2000, 2000, 2000],
         'TotalWorkingYears': [5, 10, 4, 7, 10, 10],
         'TrainingTimesLastYear': [2,3,1,2,3,3],
         'WorkLifeBalance': [1, 2, 3, 3, 3, 3],
         'YearsAtCompany': [5, 10, 4, 7, 10, 10],
         'YearsInCurrentRole': [1, 3, 1, 2, 3, 3],
         'YearsSinceLastPromotion': [3, 5, 2, 4, 5, 5],
         'YearsWithCurrManager': [2, 4, 1, 3, 4, 4],
         'JobSatisfaction': [2, 4, 1, 3, 3, 3],
         'RelationshipSatisfaction': [2, 4, 1, 3, 3, 3],
         'EnvironmentSatisfaction': [2, 4, 1, 3, 3, 3],
}

st.set_page_config(page_title="Employee Churn Prediction", page_icon=":bar_chart:", layout="wide")
st.title("Employee Churn Prediction")
st.subheader("Predict Employee Churn using DNN Architecture")
st.write("This is a simple Machine Learning Web App to predict employee churn and retain your best employees")

# Create DataFrame
sample = pd.DataFrame(sample)

def App1():
    col1, col2 = st.columns(2)
    with col1.container():
        with st.container(500):
            st.write("Input your employees")
            # Define Unique Key for sliders
            age_slider_key = "age_slider"
            JobInvl_slider_key = "job_involvement_slider"
            Perf_slider_key = "performance_slider"
            Income_slider_key = "income_slider"
            Ttl_slider_key = "total_working_years_slider"
            Train_slider_key = "training_times_slider"
            Wlb_slider_key = "work_life_balance_slider"
            YearsAtComp_slider_key = "years_at_company_slider"
            YearsCurrRole_slider_key = "years_in_current_role_slider"
            YearsLastProm_slider_key = "years_since_last_promotion_slider"
            YearsCurrManag_slider_key = "years_with_current_manager_slider"
            Envsat_slider_key = "environment_satisfaction_slider"
            JobSat_slider_key = "job_satisfaction_slider"
            RelSat_slider_key = "relationship_satisfaction_slider"
                
            # Employee data input fields
            e1 = st.slider("Age (Range 18 - 60)", 18, 60, 30, key= age_slider_key)
            e2 = st.radio("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
            e3 = st.radio("Department Name", ['Corporate Functions', 'Delivery', 'HR', 'Marketing', 'Product', 'Sales'])
            e4 = st.radio("Education Field", ['Diploma', 'Bachelors', 'Masters', 'Doctorate'])
            e5 = st.radio("Gender", ['Female', 'Male'])
            e6 = st.radio("Marital Status", ['Single', 'Married', 'Divorced'])
            e7 = st.radio("Job Level", ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'])
            e8 = st.radio("OverTime", ['No', 'Yes'])
            e9 = st.slider("Job Involvement", 1, 4, 2, key= JobInvl_slider_key)
            e10 = st.slider("Performance Score", 1, 5, 3, key= Perf_slider_key)
            e11 = st.slider("Monthly Income in $", 1000, 20000, 5000, key= Income_slider_key)
            e12 = st.slider("Total Working Years", 0, 40, 20, key= Ttl_slider_key)
            e13 = st.slider("Training Times Last Year", 0, 6, 2, key= Train_slider_key)
            e14 = st.slider("Work Life Balance", 1, 4, 2, key= Wlb_slider_key)
            e15 = st.slider("Years at Company", 1, 15, 7, key= YearsAtComp_slider_key)
            e16 = st.slider("Years In Current Role", 1, 7, 2, key= YearsCurrRole_slider_key)
            e17 = st.slider("Years Since Last Promotion", 1, 6, 3, key= YearsLastProm_slider_key)
            e18 = st.slider("Years With Current Manager", 1, 6, 3, key= YearsCurrManag_slider_key)
            e19 = st.slider("Job Satisfaction", 1, 4, 2, key= JobSat_slider_key)
            e20 = st.slider("Relationship Satisfaction", 1, 4, 2, key= RelSat_slider_key)
            e21 = st.slider("Environment Satisfaction", 1, 4, 2, key= Envsat_slider_key)
                
            # Convert categorical features to numerical
            e2 = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}[e2]
            e3 = {'Corporate Functions': 0, 'Delivery': 1, 'HR': 2, 'Marketing': 3, 'Product': 4, 'Sales': 5}[e3]
            e4 = {'Diploma': 1, 'Bachelors': 0, 'Masters': 3, 'Doctorate': 2}[e4]
            e5 = {'Female': 0, 'Male': 1}[e5]
            e6 = {'Single': 2, 'Married': 1, 'Divorced': 0}[e6]
            e7 = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4, 'L6': 5, 'L7': 6}[e7]
            e8 = {'No': 0, 'Yes': 1}[e8]
                
            user_input_values = {'Age': e1,
                                'BusinessTravel': e2,
                                'Department': e3,
                                'EducationField': e4,
                                'Gender': e5,
                                'MaritalStatus': e6,
                                'JobLevel': e7,
                                'OverTime': e8,
                                'JobInvolvement': e9,
                                'PerformanceRating': e10,
                                'MonthlyIncome': e11,
                                'TotalWorkingYears': e12,
                                'TrainingTimesLastYear': e13,
                                'WorkLifeBalance': e14,
                                'YearsAtCompany': e15,
                                'YearsInCurrentRole': e16,
                                'YearsSinceLastPromotion': e17,
                                'YearsWithCurrManager': e18,
                                'JobSatisfaction': e19,
                                'RelationshipSatisfaction': e20,
                                'EnvironmentSatisfaction': e21,
            }

            # Convert the dictionary to a pandas DataFrame
            input_df = pd.DataFrame.from_dict(user_input_values, orient='index').T

            # Prepare input tensor
            input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

            # Ensure model is in evaluation mode
            model.eval()

    with col2.container():
        st.write("Dataframe Preview")
        st.dataframe(input_df)

        # Create a button to trigger prediction
        if st.button("Predict"):
            # Preprocess user input
            with st.spinner('Wait for it...'):
                time.sleep(5)
            
            preprocessed_input = preprocess_input(input_tensor)

            # Make prediction
            prediction = predict_employee_churn(preprocessed_input)

            # Display prediction result
            if prediction == 1:
                st.write("Your employee may leave the company. (╥﹏╥)")
            else:
                st.write("Your employee may stay in the company. ⸜(｡ ˃ ᵕ ˂ )⸝♡")   

def App2():
    st.text("Download this sample for testing your the model below")
    st.dataframe(sample)
    # Create a button to go to link
    # Define the URL
    url = "https://www.kaggle.com/datasets/jash312/hr-employee-attrition-datasets?select=HR+Employee+data.csv"

    # Create a clickable link using markdown
    #st.markdown(f"[Link to HR Employee Attrition Dataset]({url})")
    # Button to upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Load the model
            model = torch.load(PATH)
            model.eval()
                
            # Read the CSV file into a pandas dataframe
            data = pd.read_csv(uploaded_file)

            # Get a list of actual column names
            actual_columns = data.columns

            # Define a list of column names to exclude (if they exist)
            columns_to_exclude = [col for col in ['Unnamed: 0'] if col in actual_columns]

            # Check if columns need exclusion
            if columns_to_exclude:
                data = data.drop(columns_to_exclude, axis=1)

            # Display the processed dataframe
            st.write("Dataframe Preview:")
            st.dataframe(data.head())

            # Preprocess Input Data
            # Convert DataFrame to NumPy array
            input_tensor = data.values
                
            # Convert NumPy array to tensor
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32) # Preprocess data directly to tensor

            # Make Predictions
            threshold = 0.5
            with torch.no_grad():
                preds = model(input_tensor)
                y_pred = ["Your employee may leave the company. (╥﹏╥)" if pred > threshold else "Your employee may stay in the company. ⸜(｡ ˃ ᵕ ˂ )⸝♡" for pred in preds]

            data['Predicted_target'] = y_pred

            st.write("Processed Data:")
            st.dataframe(data)
                
        except Exception as e:
            st.error(f"Failed to process data: {e}")

# Define function to preprocess user input
def preprocess_input(input_tensor):# Convert tensor to DataFrame
    input_df = pd.DataFrame(input_tensor.numpy(), columns=['Age', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'JobLevel', 'OverTime', 'JobInvolvement', 'PerformanceRating', 'MonthlyIncome', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobSatisfaction', 'RelationshipSatisfaction', 'EnvironmentSatisfaction'])

    # Initialize MinMaxScaler
    min_max_scaler = MinMaxScaler()
    # Perform min-max normalization on numerical features
    numerical_features = ['Age', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'JobLevel', 'OverTime', 'JobInvolvement', 'PerformanceRating', 'MonthlyIncome', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobSatisfaction', 'RelationshipSatisfaction', 'EnvironmentSatisfaction']
    input_df[numerical_features] = min_max_scaler.fit_transform(input_df[numerical_features])

    # Convert DataFrame back to tensor
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

    return input_tensor

# Define function to make predictions
def predict_employee_churn(input_data):
    with torch.no_grad():
        output = model(input_data)
        prediction = torch.round(torch.sigmoid(output)).item()
        return prediction

c = st.container()
tab1, tab2 = st.tabs(["Single Prediction", "Predict Data using uploaded CSV"])
with tab1.container(): 
    App1()
with tab2.container():
    App2()

