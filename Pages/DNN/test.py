from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Optional
import pandas as pd

class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    MaritalStatus: str
    JobLevel: str
    OverTime: str
    JobInvolvement: int
    PerformanceRating: int
    MonthlyIncome: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    JobSatisfaction: int
    RelationshipSatisfaction: int
    EnvironmentSatisfaction: int

# Replace the manual input parsing with Pydantic model validation
user_input_values1 = st.form_submit_button("Single Prediction")
input_data1 = EmployeeInput(**user_input_values1)