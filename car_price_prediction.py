# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pickle

# Function to transform the dataset
def transform_dataset(df):
    distance_mapping = {
        'distance_Low': 'Low',
        'distance_Medium': 'Medium',
        'distance_High': 'High',
        'distance_Very High': 'Very High',
        'distance_Extremely High': 'Extremely High'
    }
    
    fuel_mapping = {
        'fuel_CNG': 'CNG',
        'fuel_Diesel': 'Diesel',
        'fuel_LPG': 'LPG',
        'fuel_Petrol': 'Petrol'
    }

    seller_mapping = {
        'seller_Dealer': 'Dealer',
        'seller_Individual': 'Individual',
        'seller_Trustmark Dealer': 'Trustmark Dealer'
    }

    owner_mapping = {
        'owner_First Owner': 'First Owner',
        'owner_Fourth & Above Owner': 'Fourth & Above Owner',
        'owner_Second Owner': 'Second Owner',
        'owner_Third Owner': 'Third Owner'
    }

    car_brand_mapping = {
        'car_brand_Chevrolet': 'Chevrolet',
        'car_brand_Ford': 'Ford',
        'car_brand_Honda': 'Honda',
        'car_brand_Hyundai': 'Hyundai',
        'car_brand_Mahindra': 'Mahindra',
        'car_brand_Maruti': 'Maruti',
        'car_brand_Renault': 'Renault',
        'car_brand_Tata': 'Tata',
        'car_brand_Toyota': 'Toyota',
        'car_brand_Volkswagen': 'Volkswagen'
    }

    def get_column_name(row, mapping):
        for col in mapping:
            if row[col] == 1:
                return mapping[col]
        return None

    df['distance'] = df.apply(lambda row: get_column_name(row, distance_mapping), axis=1)
    df['fuel'] = df.apply(lambda row: get_column_name(row, fuel_mapping), axis=1)
    df['seller_type'] = df.apply(lambda row: get_column_name(row, seller_mapping), axis=1)
    df['owner'] = df.apply(lambda row: get_column_name(row, owner_mapping), axis=1)
    df['car_brand'] = df.apply(lambda row: get_column_name(row, car_brand_mapping), axis=1)
    df['transmission'] = df['transmission'].map({1: 'Automatic', 0: 'Manual'})

    return df[['car_brand', 'mileage_km', 'engine', 'seats', 'car_age', 'transmission', 'fuel', 'seller_type', 'owner', 'distance']]

# Title
st.title("Car Price Prediction App")
st.write("This app predicts the price of a car based on its features.")

# Load data
df = pd.read_csv('processed_cardekho.csv')
df = transform_dataset(df)

# Display data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Create selection arrays
car_brandArr = df['car_brand'].unique()
transmissionArr = df['transmission'].unique()
fuelArr = df['fuel'].unique()
seller_typeArr = df['seller_type'].unique()
ownerArr = df['owner'].unique()
distanceArr = df['distance'].unique()

# User input for new prediction
st.sidebar.header("Input Features")
car_brand = st.sidebar.selectbox("Car Brand", car_brandArr)
mileage_km = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=500000)
engine = st.sidebar.number_input("Engine (cc)", min_value=500, max_value=5000)
seats = st.sidebar.number_input("Seats", min_value=2, max_value=8)
car_age = st.sidebar.number_input("Car Age", min_value=0, max_value=30)
transmission = st.sidebar.selectbox("Transmission", transmissionArr)
fuel = st.sidebar.selectbox("Fuel Type", fuelArr)
seller_type = st.sidebar.selectbox("Seller Type", seller_typeArr)
owner = st.sidebar.selectbox("Owner Type", ownerArr)
distance = st.sidebar.selectbox("Distance Category", distanceArr)

# Prepare input for prediction
input_features = pd.DataFrame({
    'car_brand': [car_brand],
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
input_features = pd.get_dummies(input_features)
df_encoded = pd.get_dummies(df.drop(columns=['selling_price']))
input_features = input_features.reindex(columns=df_encoded.columns, fill_value=0)

# Load scaler and scale input features
scaler = StandardScaler()
X = df.drop(columns=['selling_price'])
scaler.fit(X)
input_features_scaled = scaler.transform(input_features)

# Load model
with open('catboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make prediction
prediction = model.predict(input_features_scaled)

# Display prediction
st.subheader("Predicted Selling Price")
st.write(f"${prediction[0]:,.2f}")
