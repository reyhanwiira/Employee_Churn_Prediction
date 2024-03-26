import numpy as np
import pandas as pd
import re
import time
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

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
    
def scaling(input_df):
    # Convert tensor to DataFrame
    input_df = pd.DataFrame(input_tensor.numpy(), columns=["Age", "BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "JobLevel", "OverTime", "JobInvolvement", "PerformanceRating", "MonthlyIncome", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "JobSatisfaction", "RelationshipSatisfaction", "EnvironmentSatisfaction"])

    # Initialize MinMaxScaler
    min_max_scaler = MinMaxScaler()
    # Perform min-max normalization on numerical features
    numerical_features = ["Age", "BusinessTravel", "Department", "EducationField", "Gender", "MaritalStatus", "JobLevel", "OverTime", "JobInvolvement", "PerformanceRating", "MonthlyIncome", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "JobSatisfaction", "RelationshipSatisfaction", "EnvironmentSatisfaction"]
    input_df[numerical_features] = min_max_scaler.fit_transform(input_df[numerical_features])

    # Convert DataFrame back to tensor
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

def preprocess_input(input_data: EmployeeInput) -> torch.Tensor:
    input_df = input_data.dict()
    # Your existing preprocessing logic here...

    return input_tensor

def predict_employee_churn(input_data: EmployeeInput) -> int:
    # Your prediction logic here...