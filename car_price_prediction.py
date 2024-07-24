# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pickle

# Title
st.title("Car Price Prediction App")
st.write("This app predicts the price of a car based on its features.")

# Load data
df = pd.read_csv('cardekho.csv')

# Display data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Create selection Arrays
# car_brandArr = df['car_brand'].unique()
transmissionArr = df['transmission'].unique()
fuelArr = df['fuel'].unique()
seller_typeArr = df['seller_type'].unique()
ownerArr = df['owner'].unique()

# User input for new prediction
st.sidebar.header("Input Features")
# car_brand = st.sidebar.selectbox("Car Brand", car_brandArr)
mileage_km = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=500000)
engine = st.sidebar.number_input("Engine (cc)", min_value=500, max_value=5000)
seats = st.sidebar.number_input("Seats", min_value=2, max_value=8)
car_age = st.sidebar.number_input("Car Age", min_value=0, max_value=30)
transmission = st.sidebar.selectbox("Transmission", transmissionArr)
fuel = st.sidebar.selectbox("Fuel Type", fuelArr)
seller_type = st.sidebar.selectbox("Seller Type", seller_typeArr)
owner = st.sidebar.selectbox("Owner Type", ownerArr)
distance = st.sidebar.selectbox("Distance Category", ['Low', 'Medium', 'High', 'Very High', 'Extremely High'])

# Prepare input for prediction
input_features = pd.DataFrame({                 
    # 'car_brand': [car_brand],
    'mileage_km': [mileage_km],
    'engine': [engine],
    'seats': [seats],
    'car_age': [car_age],
    'transmission': [transmission],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'owner': [owner],
    'distance': [distance]
})

# One-hot encode categorical features (dummy encoding)
input_features_encoded = pd.get_dummies(input_features)
df_encoded = pd.get_dummies(df.drop(columns=['selling_price']))
input_features_encoded = input_features_encoded.reindex(columns=df_encoded.columns, fill_value=0)

# Load scaler and scale input features
scaler = StandardScaler()
X = df.drop(columns=['selling_price'])
X_encoded = pd.get_dummies(X)
scaler.fit(X_encoded)
input_features_scaled = scaler.transform(input_features_encoded)

# Load model
with open('catboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make prediction
prediction = model.predict(input_features_scaled)

# Display prediction
st.subheader("Predicted Selling Price")
st.write(f"${prediction[0]:,.2f}")
