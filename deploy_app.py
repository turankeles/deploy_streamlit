import streamlit as st 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import pickle

data=pd.read_csv("final_scout_not_dummy.csv")
data=data[data.make_model!="Audi A2"]
data=data[data.make_model!="Renault Duster"]

car_counts=st.selectbox("Plots",["Car Ages","Price","Consumption"])
model=pickle.load(open("model","rb"))
lr_model=pickle.load(open("lr_model","rb"))

col_transformer=pickle.load(open("col_transformer","rb"))
if car_counts=="Car Ages":
    st.subheader("Average Ages")
    st.bar_chart(data.groupby("make_model").mean()["age"])
        
if car_counts=="Price":
    st.subheader("Average Prices")
    st.bar_chart(data.groupby("make_model").mean()["price"])
if car_counts=="Consumption":    
    st.subheader("Average gas consumption")
    st.bar_chart(data.groupby("make_model").mean()["cons_comb"])


age=st.sidebar.slider("What is the age of your car:",0,4,step=1)
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
consumption=st.sidebar.slider("What is the mpg of your car", 1.0,10.0, step=0.1)
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))

my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'cons_comb':consumption,
    "make_model": car_model
    
}

df=pd.DataFrame([my_dict])
st.table(df)

new_df=col_transformer.transform(pd.DataFrame([my_dict]))

if st.button("Predict Random Forest"):
    a=model.predict(new_df)
  
    st.success(f"The estimated price is : $ {int(a)}")

if st.button("Predict with XGBoost") :
    a=lr_model.predict(new_df)
  
    st.success(f"The estimated price is : $ {int(a)}")
    
    