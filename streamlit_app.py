import streamlit as st
import numpy as np
import pandas as pd

#title and description
st.title('😁 Loan Eligability System')

st.write('this is a web application for a machine learning model for a loan eligability system')

#expanders to display dataset info
with st.expander("Data Set"):
  st.write("**Here's the raw data**")
  df = pd.read_csv('train.csv')
  df.dropna(axis=0 , inplace = True)
  df

# the x an y variables
  st.write("**X-Variables**")
  x_raw = df.drop("Loan_Status", axis=1)
  x_raw.drop("Loan_ID", axis=1 , inplace=True)
  x_raw.drop("Loan_Amount_Term", axis=1 , inplace=True)
  x_raw

  st.write("**Y-Variable**")
  y_raw = df.Loan_Status
  y_raw

with st.expander("data visualization"):
  st.write("**Scatter Plot of income vs loan amount")
  st.scatter_chart(data = df, x = "ApplicantIncome" , y = "LoanAmount", color = "Loan_Status")

#input features
with st.sidebar:
  st.header("Applicant details (input)")
  Gender = st.selectbox("Gender",("Male","Female"))
  Married = st.selectbox("Married",("Yes","No"))
  Dependants = st.selectbox("Dependants",("0","1","2","3+"))
  Education = st.selectbox("Education", ("Graduate","Not Graduate"))
  Self_Employed = st.selectbox("Self Employed", ("Yes","No"))
  Credit_History = st.selectbox("Credit history", ("0","1"))
  Property_Area = st.selectbox("Property Area", ("Urban","Rural","Semiurban"))
  ApplicantIncome = st.slider("Applicant Income",0,50000,81000)
  CoapplicantIncome = st.slider("Co-Applicant Income",0,50000,20000)
  LoanAmount = st.slider("Loan amount", 17 , 100, 600)
  
  
  
