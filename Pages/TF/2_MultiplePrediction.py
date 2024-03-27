import streamlit as st
import pandas as pd
import pickle
import time
import requests
import base64

st.set_page_config(page_title="Multiple Prediction", page_icon="", layout="wide")

st.title("Predict your Employee Churn using Transformer Pipeline")
st.write("This is a simple Machine Learning Web App to predict employee churn and retain your best employees")
st.write("Disclaimer: Don't rely too much on these results, expert judgment is still needed to see any invisible influences")

row1 = st.columns(1)
row2 = st.columns(1)
row3 = st.columns(1)
row4 = st.columns(1)

with row1[0]:
    st.subheader("Multiple Prediction")
with row2[0]:

    df = pd.DataFrame(columns=[
        "Age", "BusinessTravel", "Department", "EducationField", "Gender",
        "MaritalStatus", "JobLevel", "OverTime", "JobInvolvement",
        "PerformanceRating", "MonthlyIncome", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
        "JobSatisfaction", "RelationshipSatisfaction", "EnvironmentSatisfaction"
        ])

    editable_df = st.data_editor(df, num_rows="dynamic",
        column_config={
            "Age": st.column_config.NumberColumn(
                "Age",
                help="Input employee ages",
                min_value=18, max_value=60, step=1, default= 28, required=True),
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
                min_value=1, max_value=4, step=1, default= 2, required=True),
            "PerformanceRating": st.column_config.NumberColumn(
                "Performance Score",
                help="Input employees Performance Score",
                min_value=1, max_value=4, step=1, default= 2, required=True),
            "MonthlyIncome": st.column_config.NumberColumn(
                "Monthly Income",
                help="Input employees monthly income",
                min_value=1000, max_value=20000, step=1, default= 2000, required=True),
            "TotalWorkingYears": st.column_config.NumberColumn(
                "Total Working Years",
                help="Input employees Total Working Years ",
                min_value=1, max_value=40, step=1, default= 1, required=True),
            "TrainingTimesLastYears": st.column_config.NumberColumn(
                "Training Times Last Years",
                help="Input employees training times last years ",
                min_value=0, max_value=6, step=1, default= 1, required=True),
            "WorkLifeBalance": st.column_config.NumberColumn(
                "Work Life Balance",
                help="Input work life balance score",
                min_value=1, max_value=4, step=1, default= 2, required=True),
            "YearsAtCompany": st.column_config.NumberColumn(
                "Years At Company",
                help="Input employees years at company",
                min_value=1, max_value=15, step=1, default= 2, required=True),
            "YearsInCurrentRole": st.column_config.NumberColumn(
                "Years In Current Role",
                help="Input employees years in current role",
                min_value=1, max_value=7, step=1, default= 2, required=True),
            "YearsSinceLastPromotion": st.column_config.NumberColumn(
                "Years Since Last Promotion",
                help="Input employees years since last promotion",
                min_value=1, max_value=6, step=1, default= 1, required=True),
            "YearsWithCurrManager": st.column_config.NumberColumn(
                "Years with Current Manager",
                help="Input employees years with current manager",
                min_value=1, max_value=6, step=1, default= 2, required=True),
            "JobSatisfaction": st.column_config.NumberColumn(
                "Job Satisfaction",
                help="Input employees job satisfaction",
                min_value=1, max_value=4, step=1, default= 3, required=True),
            "RelationshipSatisfaction": st.column_config.NumberColumn(
                "Relationship Satisfaction",
                help="Input employees relationship satisfaction",
                min_value=1, max_value=4, step=1, default= 3, required=True),
            "EnvironmentSatisfaction": st.column_config.NumberColumn(
                "Environment Satisfaction",
                help="Input employees environment satisfaction",
                min_value=1, max_value=4, step=1, default= 3, required=True),
        },
        hide_index=True,
    )

    st.write("Dataframe Preview")
    st.dataframe(editable_df)
    uploaded_file = pd.DataFrame(editable_df)

    def process_data(data):
        # Load the pre-trained model
        with open('./Models/pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)

        # Check if all required columns are present
        required_columns = ['Age', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 
                            'MaritalStatus', 'JobLevel', 'OverTime', 'JobInvolvement',
                            'PerformanceRating', 'MonthlyIncome', 'TotalWorkingYears',
                            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                            'JobSatisfaction', 'RelationshipSatisfaction', 'EnvironmentSatisfaction']
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Columns are missing: {missing_columns}")

        # Manually encode the column
        data['Gender'] = editable_df['Gender'].map({'Female': 0, 'Male': 1})
        data['OverTime'] = editable_df['OverTime'].map({'No': 0, 'Yes': 1})

        # Exclude specific columns
        excluded_columns = ['Attrition', 
                            'LeavingYear',
                            'Reason',
                            'RelievingStatus']  # Example: Columns to be excluded
        data = data.drop(columns=excluded_columns, errors='ignore')

        # Perform predictions
        result = pipeline.predict(data)
                
        # Assign predictions based on result
        y_pred = ["An employee may leave the organization." if pred == 1 
                else "An employee may stay with the organization." for pred in result]
                    
        # Add predicted target to the data
        data['Output'] = y_pred

        return data

    # Create a button to trigger prediction
    if st.button("Predict", key="predictApp2"):
        # Preprocess user input
        with st.spinner("Processing data..."):
            time.sleep(2)

            # Access all rows of the DataFrame and convert them to a list of dictionaries
            user_input_values_list = editable_df.to_dict(orient='records')

            # Initialize an empty list to store processed data
            processed_data_list = []

            # Iterate over each row and process them individually
            for user_input_values in user_input_values_list:
                try:
                    # Convert the dictionary to a DataFrame
                    user_input_df = pd.DataFrame([user_input_values])

                    # Process the data
                    processed_data = process_data(user_input_df)

                    # Append the processed data to the list
                    processed_data_list.append(processed_data)

                except Exception as e:
                    st.error(f"Failed to process data: {e}")

            if processed_data_list:
                # Concatenate processed data from all rows
                processed_data_df = pd.concat(processed_data_list)
                st.write("Processed Data:")
                st.dataframe(processed_data_df)
            else:
                st.error("No data processed.")

with row3[0]:
    st.subheader("Multiple Prediction with CSV")
with row4[0]:
    # Button to upload CSV file
    st.text("If you tired by inputing data manually, you can upload your CSV file here.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # if uploaded_file is not None:
    #     try:
    #         # Load data from CSV
    #         data = pd.read_csv(uploaded_file)
    #         data = data.drop_duplicates()
    #         # Rename JobLevel_updated to JobLevel
    #         data = data.rename(columns={'JobLevel_updated': 'JobLevel'})

    #         # Process the data
    #         processed_data = process_data(data)
            
    #         # Save the processed data to a CSV file
    #         st.write("Processed Data:")
    #         st.write(processed_data)
    #         st.write("Saving the processed data...")
    #         processed_data.to_csv('processed_data.csv', index=False)
    #         st.success("Data saved successfully!")
    #     except Exception as e:
    #         st.error(f"Failed to open file: {e}")

    if uploaded_file is not None:
        try:
            # Load data from CSV
            data = pd.read_csv(uploaded_file)
            data = data.drop_duplicates()
            # Rename JobLevel_updated to JobLevel
            data = data.rename(columns={'JobLevel_updated': 'JobLevel'})

            # Process the data
            processed_data = process_data(data)
            
            # Count the number of employees predicted to leave
            leave_count = processed_data['Output'].value_counts().get("An employee may leave the organization.", 0)
            total_count = len(processed_data)
            
            st.write("Processed Data:")
            st.write(processed_data)
            
            st.write(f"Out of {total_count} employees:")
            st.write(f"- {leave_count} employees may leave the organization.")
            st.write(f"- {total_count - leave_count} employees may stay with the organization.")

            # Filter the processed data for employees predicted to leave
            leave_data = processed_data[processed_data['Output'] == "An employee may leave the organization."]

            # Count occurrences of low values in specified columns
            low_worklife_balance_count = (leave_data['WorkLifeBalance'] == 1).sum()
            low_job_satisfaction_count = (leave_data['JobSatisfaction'] == 1).sum()
            low_environment_satisfaction_count = (leave_data['EnvironmentSatisfaction'] == 1).sum()
            low_relationship_satisfaction_count = (leave_data['RelationshipSatisfaction'] == 1).sum()

            # Display the counts
            st.write("Counts of employees predicted to leave with low WorkLifeBalance, Job, Environment, and Relationship Satisfaction:")
            st.write(f"- Low WorkLifeBalance: {low_worklife_balance_count} employees")
            st.write(f"- Low Job Satisfaction: {low_job_satisfaction_count} employees")
            st.write(f"- Low Environment Satisfaction: {low_environment_satisfaction_count} employees")
            st.write(f"- Low Relationship Satisfaction: {low_relationship_satisfaction_count} employees")

            # Provide recommendations based on the leave count
            if leave_count > 0:
                st.write("Recommendations:")
                st.write("- Conduct exit interviews to understand reasons for leaving.")
                st.write("- Implement retention strategies such as improving work-life balance, offering career development opportunities, etc.")
            else:
                st.write("No employees predicted to leave. No specific recommendations at this time.")

            st.write("Saving the processed data...")
            processed_data.to_csv('processed_data.csv', index=False)
            st.success("Data saved successfully!")
        except Exception as e:
            st.error(f"Failed to open file: {e}")

