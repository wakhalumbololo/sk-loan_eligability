import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#title and description
st.title('😁 Loan Eligability System')

st.write('this is a web application for a machine learning model for a loan eligability system')

#expanders to display dataset info
with st.expander("Data Set"):
  st.write("**Here's the raw data**")
  df = pd.read_csv('train.csv')
  df.dropna(axis=0 , inplace = True)
  df["Loan_Status"].replace({"Y": "Loan Approved" , "N" : "Rejected"})
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
  Dependents = st.selectbox("Dependants",("0","1","2","3+"))
  Education = st.selectbox("Education", ("Graduate","Not Graduate"))
  Self_Employed = st.selectbox("Self Employed", ("Yes","No"))
  Credit_History = st.slider("Credit history", 0,1,1)
  Property_Area = st.selectbox("Property Area", ("Urban","Rural","Semiurban"))
  ApplicantIncome = st.slider("Applicant Income",0,50000,81000)
  CoapplicantIncome = st.slider("Co-Applicant Income",0,50000,20000)
  LoanAmount = st.slider("Loan amount", 17 , 100, 600)

#creating a dataframe for the input features
data = {"Gender":Gender,
        "Married":Married,
        "Dependents":Dependents,
        "Education":Education,
        "Self_Employed":Self_Employed,
        "Credit_History":Credit_History,
        "Property_Area":Property_Area,
        "ApplicantIncome":ApplicantIncome,
        "CoapplicantIncome":CoapplicantIncome,
        "LoanAmount":LoanAmount}
            
input_df = pd.DataFrame(data , index = [0])
input2_df = pd.concat([input_df, x_raw] , axis = 0)

with st.expander("Applicant Details"):
  st.write("**Input data**")
  input_df
  st.write("**Combined Input data**")
  input2_df
  
#DATA PREPERATIONS FOR MACHINE LEARNING 
# ENCODING X VALUES
encode = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
input3_df = pd.get_dummies(input2_df , prefix = encode)
x_train = input3_df[1:]
input_row = input3_df[:1]

#ENCODING Y VALUES
encode2 = {"Y":0, "N":1}
def target_encode(value):
  return encode2[value]

y_train = y_raw.apply(target_encode)

with st.expander("training data"):
  st.write("**input row**")
  input_row
  st.write("**X Values**")
  x_train
  st.write("**Y Values**")
  y_train

# model training
clf = RandomForestClassifier()
clf.fit(x_train,y_train)

prediction = clf.predict(input_row)
pred_proba = clf.predict_proba(input_row)

df_pred_proba = pd.DataFrame(pred_proba)
df_pred_proba.columns = ["Loan Approved","Rejected"]
df_pred_proba.rename(columns = {0:"Loan Approved",
                                1:"Rejected"
                               })

#displaying the prediction
st.subheader("Loan Application Status")
st.dataframe(df_pred_proba ,
             column_config ={
               "Loan Approved" : st.column_config.ProgressColumn(
                 "Loan Approved",
                 format = "%f",
                 width="200",
                 min_value=0,
                 max_value=1),
               "Rejected" : st.column_config.ProgressColumn(
                 "Rejected",
                 format = "%f",
                 width="200",
                 min_value=0,
                 max_value=1),
             },hide_index = True)   

result = np.array(["Loan Approved" ,"Rejected"])
st.success(str(result[prediction][0]))

