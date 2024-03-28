<h1 align="center">Employee Churn Prediction App</h1>
<div align= "center"><img src="" />
  <h4>This is a simple Machine Learning Web App to predict employee churn and retain your best employees using transformer pipeline approach.</h4>
</div>

# Employee-Churn-Prediction-App

## :bookmark_tabs:Table of contents
* [General info](#general-info)
* [Development](#development)
* [Technologies](#technologies)
* [Setup](#setup)

## :scroll: General info
### Motivation
In today's competitive job market, retaining top talent is crucial. But many organizations face a hidden enemy: Employee Churn. This refers to the rate at which employees leave a company, creating a costly and disruptive employee life cycle elsewhere.
### What is employee churn prediction app?
An employee churn prediction app using transformer pipeline approach, Random Forest Classifier for precise that can help a company or employer to predict and retain their best employee. This app uses several important parameter or feature that can affect employee to leave or not to leave a company. Such as Salary, Career Path, How much they get training in their company, Company Culture, Job Satisfaction, Work life balance, etc. Those can affect employee performance, if their performance is decrease significant its can affect a whole team or a company. 

## :computer:Development
### Model developement
The web app is built using Random Forest algorithm. Random Forest is a versatile machine learning algorithm that can be used for both classification and regression tasks. Its architecture is based on the concept of ensemble learning, where multiple models are combined to improve the overall performance. Random Forest is constructed from a collection of decision trees, hence the term "forest". The architecture of Random Forests makes them robust and less prone to overfitting compared to individual decision trees. By combining multiple trees and introducing randomness through bootstrapping and feature selection, Random Forests can effectively handle a variety of datasets and achieve high accuracy in predictions.

### Dataset
I use HR Analytics Dataset. HR Analytics helps us with interpreting organizational data. It finds the people-related trends in the data and allows the HR Department to take the appropriate steps to keep the organization running smoothly and profitably. Attrition in a corporate setup is one of the complex challenges that the people managers and the HRs personnel have to deal with.
dataset source:
https://www.kaggle.com/datasets/jash312/hr-employee-attrition-datasets?select=Employee_office_survey.csv

### Frontend Development
The application's front-end is made with Streamlit. Streamlit is an open source app framework in Python language. It helps to create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc. For our case the front-end is composed of three web pages. The main page is Hello.py which is a welcoming page used to introduce you to my project. The side bar on the left allows the user to navigate to the single prediction page and the multiple prediction page. In the single prediction page the user can fill information about age, department, job level, satisfaction, etc and gets a prediction based on inputed information. And for the multiple prediction its allows user to predict more than one employee that user want to predict, and get all the answer directly in a csv form. So user can download the output data. I also added some insight based on user uploaded csv. How much total employee that they have, and how much employee that predicted will leave the company and a little recommendation that can help employer / HR Department to retain their selected employee. 

## :rocket: Technologies
The project is created with:
python: 3.11.7
pandas==1.5.1 
numpy==1.20.3
scikit-learn==1.4.1.post1
streamlit==1.32.2
st-pages==0.4.4

### Use the hosted version on Streamlit Cloud
http://employee-churn-predict.streamlit.app/

