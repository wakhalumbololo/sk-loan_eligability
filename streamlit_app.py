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
  x_raw

  st.write("**Y-Variable**")
  y_raw = df.Loan_Status
  y_raw

with st.expander("data visualization"):
  st.write("**Scatter Plot of income vs loan amount")
  st.scatter_chart(data = df, x = "ApplicantIncome" , y = "LoanAmount", color = "Loan_Status")


