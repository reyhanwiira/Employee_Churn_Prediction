from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.activation = nn.ReLU()  # Changed activation to ReLU for hidden layers
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

# Load Our pre-trained model
PATH = "./Models/churn-model-full.pt"
model = torch.load(PATH)
model.eval()

st.set_page_config(page_title="Employee Churn Prediction", page_icon=":bar_chart:", layout="wide")
st.title("Employee Churn Prediction")
st.subheader("Predict Employee Churn using DNN Architecture")
st.write("This is a simple Machine Learning Web App to predict employee churn and retain your best employees") 

# Define function to make predictions
def predict_employee_churn(input_data):
    with torch.no_grad():
        output = model(input_data)
        prediction = torch.round(torch.sigmoid(output)).item()
        return prediction

row1 = st.columns(1)
row2 = st.columns(1)

with row1[0]:
    st.subheader("Multiple Prediction")
with row2[0]:
    # Define function to preprocess user input
    def preprocess_input(input_tensor):

        input_df = pd.DataFrame(input_tensor, 
                                columns=["Age",
                                         "BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "JobLevel", "OverTime", 
                                        "JobInvolvement", 
                                        "PerformanceRating", 
                                        "MonthlyIncome", 
                                        "TotalWorkingYears", 
                                        "TrainingTimesLastYear", 
                                        "WorkLifeBalance", 
                                        "YearsAtCompany", 
                                        "YearsInCurrentRole", 
                                        "YearsSinceLastPromotion", 
                                        "YearsWithCurrManager", 
                                        "JobSatisfaction", 
                                        "RelationshipSatisfaction", 
                                        "EnvironmentSatisfaction"])

        # Encode categorical variables
        categorical_features = ["BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "JobLevel", "OverTime"]
        encoded_features = pd.get_dummies(input_df[categorical_features])
        input_df = pd.concat([input_df.drop(columns=categorical_features), encoded_features], axis=1)

        # Initialize MinMaxScaler
        min_max_scaler = MinMaxScaler()

        # Perform min-max normalization on numerical features
        numerical_features = ["Age", 
                                        "JobInvolvement", 
                                        "PerformanceRating", 
                                        "MonthlyIncome", 
                                        "TotalWorkingYears", 
                                        "TrainingTimesLastYear", 
                                        "WorkLifeBalance", 
                                        "YearsAtCompany", 
                                        "YearsInCurrentRole", 
                                        "YearsSinceLastPromotion", 
                                        "YearsWithCurrManager", 
                                        "JobSatisfaction", 
                                        "RelationshipSatisfaction", 
                                        "EnvironmentSatisfaction"]
        input_df[numerical_features] = min_max_scaler.fit_transform(input_df[numerical_features])

        # Convert all columns to float32
        input_df = input_df.astype('float32')

        # Convert DataFrame back to tensor
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

        return input_tensor

    st.markdown("The prediction result is based on the model that was trained on the original dataset.")
    st.text("Edit the dataframe below to predict whether your employee will leave or stay in the company.")
    df = pd.DataFrame(columns=["Age", 
         "BusinessTravel", 
         "Department", 
         "EducationField",
         "Gender",
         "MaritalStatus",
         "JobLevel",
         "OverTime",
         "JobInvolvement",
         "PerformanceRating",
         "MonthlyIncome",
         "TotalWorkingYears",
         "TrainingTimesLastYear",
         "WorkLifeBalance",
         "YearsAtCompany",
         "YearsInCurrentRole",
         "YearsSinceLastPromotion",
         "YearsWithCurrManager",
         "JobSatisfaction",
         "RelationshipSatisfaction",
         "EnvironmentSatisfaction",
    ]
    )
    editedable_df = st.data_editor(
        df, 
        num_rows="dynamic",
        column_config={
            "Age": st.column_config.NumberColumn(
                "InputAge",
                help="Input employee ages",
                min_value=18, max_value=60, step=1, default= 28),
            "BusinessTravel": st.column_config.SelectboxColumn(
                "Business Travel",
                help="Input employees business travel frequency",
                options=['Non-Travel', 
                         'Travel_Rarely', 
                         'Travel_Frequently'
                         ],required=True
                         ), 
            "Department": st.column_config.SelectboxColumn(
                "Department",
                help="Input employees department",
                options=['Corporate Functions', 
                         'Delivery', 
                         'HR', 
                         'Marketing', 
                         'Product', 
                         'Sales'
                         ],required=True
                         ), 
            "EducationField": st.column_config.SelectboxColumn(
                "Education Field",
                help="Input employees education field",
                options=['Diploma', 
                         'Bachelors', 
                         'Masters', 
                         'Doctorate'
                         ],required=True
                         ),  
            "Gender": st.column_config.SelectboxColumn(
                "Gender",
                help="Input employees gender",
                options=['Female', 
                         'Male'
                         ],required=True
                         ),  
            "MaritalStatus": st.column_config.SelectboxColumn(
                "Marital Status",
                help="Input employees marital status",
                options=['Single', 
                         'Married', 
                         'Divorced'
                         ],required=True
                         ),  
            "JobLevel": st.column_config.SelectboxColumn(
                "Job Level",
                help="Input employees job level",
                options=['L1', 
                         'L2', 
                         'L3', 
                         'L4', 
                         'L5', 
                         'L6', 
                         'L7'
                         ],required=True
                         ),  
            "OverTime": st.column_config.SelectboxColumn(
                "Over Time",
                help="Input employees over time",
                options=['No',
                         'Yes'
                         ],required=True
                         ),  
            "JobInvolvement": st.column_config.NumberColumn(
                "Job Involvement",
                help="Input employees job involvement",
                min_value=1, max_value=4, step=1, default= 2),
            "PerformanceRating": st.column_config.NumberColumn(
                "Performance Score",
                help="Input employees Performance Score",
                min_value=1, max_value=4, step=1, default= 2),
            "MonthlyIncome": st.column_config.NumberColumn(
                "Monthly Income",
                help="Input employees monthly income",
                min_value=1000, max_value=20000, step=1, default= 2000),
            "TotalWorkingYears": st.column_config.NumberColumn(
                "Total Working Years",
                help="Input employees Total Working Years ",
                min_value=1, max_value=40, step=1, default= 1),
            "TrainingTimesLastYears": st.column_config.NumberColumn(
                "Training Times Last Years",
                help="Input employees training times last years ",
                min_value=0, max_value=6, step=1, default= 0),
            "WorkLifeBalance": st.column_config.NumberColumn(
                "Work Life Balance",
                help="Input work life balance score",
                min_value=1, max_value=4, step=1, default= 1),
            "YearsAtCompany": st.column_config.NumberColumn(
                "Years At Company",
                help="Input employees years at company",
                min_value=1, max_value=15, step=1, default= 1),
            "YearsInCurrentRole": st.column_config.NumberColumn(
                "Years In Current Role",
                help="Input employees years in current role",
                min_value=1, max_value=7, step=1, default= 1),
            "YearsSinceLastPromotion": st.column_config.NumberColumn(
                "Years Since Last Promotion",
                help="Input employees years since last promotion",
                min_value=1, max_value=6, step=1, default= 1),
            "YearsWithCurrManager": st.column_config.NumberColumn(
                "Years with Current Manager",
                help="Input employees years with current manager",
                min_value=1, max_value=6, step=1, default= 1),
            "JobSatisfaction": st.column_config.NumberColumn(
                "Job Satisfaction",
                help="Input employees job satisfaction",
                min_value=1, max_value=4, step=1, default= 1),
            "RelationshipSatisfaction": st.column_config.NumberColumn(
                "Relationship Satisfaction",
                help="Input employees relationship satisfaction",
                min_value=1, max_value=4, step=1, default= 1),
            "EnvironmentSatisfaction": st.column_config.NumberColumn(
                "Environment Satisfaction",
                help="Input employees environment satisfaction",
                min_value=1, max_value=4, step=1, default= 1),
        },
        hide_index=True,
    )

    st.write("Dataframe Preview")
    st.table(editedable_df)

if st.button("Predict", key="predictApp2"):
        # Preprocess user input
        with st.spinner("Wait for it..."):
            time.sleep(5)
            
        # Access all rows of the DataFrame and convert them to a list of dictionaries
        user_input_values_list = editedable_df.to_dict(orient='records')

        # Initialize an empty list to store predictions and their corresponding outputs
        predictions = []
        output_labels = []

        # Iterate over each row and make predictions individually
        for user_input_values in user_input_values_list:
            try:
                # Convert the dictionary to a DataFrame
                user_input_df = pd.DataFrame([user_input_values])

                # Process the data
                processed_data = preprocess_input(user_input_df)

                # Make prediction
                prediction = predict_employee_churn(processed_data)

                # Assign prediction label
                output_label = "An employee may leave the organization." if prediction == 1 else "An employee may stay with the organization."

                # Append the prediction and output label to the lists
                predictions.append(prediction)
                output_labels.append(output_label)

            except Exception as e:
                st.error(f"Failed to process data: {e}")

        if predictions:
            # Add predictions and output labels to the DataFrame
            editedable_df['Prediction'] = predictions
            editedable_df['Output'] = output_labels
            st.write("Predictions:")
            st.table(editedable_df)
        else:
            st.error("No predictions made.")
