import streamlit as st
import numpy as np
import pandas as pd

#title and description
st.title('😁 Loan Eligability System')

st.write('this is a web application for a machine learning model for a loan eligability system')

#expanders to display dataset info
with st.expander("Data"):
  st.write("**Here's the raw data")
  df = pd.read_csv(https://drive.google.com/file/d/1Js7dRLUasXdvoa7CNoHe2fcoF1-sVQKf/view?usp=drive_link)
  df
