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

    return df[['selling_price', 'car_brand', 'mileage_km', 'engine', 'seats', 'car_age', 'transmission', 'fuel', 'seller_type', 'owner', 'distance']]

def reverse_transform(input_features):
    distance_mapping = {
        'Low': 'distance_Low',
        'Medium': 'distance_Medium',
        'High': 'distance_High',
        'Very High': 'distance_Very High',
        'Extremely High': 'distance_Extremely High'
    }

    fuel_mapping = {
        'CNG': 'fuel_CNG',
        'Diesel': 'fuel_Diesel',
        'LPG': 'fuel_LPG',
        'Petrol': 'fuel_Petrol'
    }

    seller_mapping = {
        'Dealer': 'seller_Dealer',
        'Individual': 'seller_Individual',
        'Trustmark Dealer': 'seller_Trustmark Dealer'
    }

    owner_mapping = {
        'First Owner': 'owner_First Owner',
        'Fourth & Above Owner': 'owner_Fourth & Above Owner',
        'Second Owner': 'owner_Second Owner',
        'Third Owner': 'owner_Third Owner'
    }

    car_brand_mapping = {
        'Chevrolet': 'car_brand_Chevrolet',
        'Ford': 'car_brand_Ford',
        'Honda': 'car_brand_Honda',
        'Hyundai': 'car_brand_Hyundai',
        'Mahindra': 'car_brand_Mahindra',
        'Maruti': 'car_brand_Maruti',
        'Renault': 'car_brand_Renault',
        'Tata': 'car_brand_Tata',
        'Toyota': 'car_brand_Toyota',
        'Volkswagen': 'car_brand_Volkswagen'
    }

    # Initialize a dictionary with zeros for all one-hot columns
    original_format = {
        'distance_Low': 0, 'distance_Medium': 0, 'distance_High': 0, 'distance_Very High': 0, 'distance_Extremely High': 0,
        'fuel_CNG': 0, 'fuel_Diesel': 0, 'fuel_LPG': 0, 'fuel_Petrol': 0,
        'seller_Dealer': 0, 'seller_Individual': 0, 'seller_Trustmark Dealer': 0,
        'owner_First Owner': 0, 'owner_Fourth & Above Owner': 0, 'owner_Second Owner': 0, 'owner_Third Owner': 0,
        'car_brand_Chevrolet': 0, 'car_brand_Ford': 0, 'car_brand_Honda': 0, 'car_brand_Hyundai': 0,
        'car_brand_Mahindra': 0, 'car_brand_Maruti': 0, 'car_brand_Renault': 0, 'car_brand_Tata': 0,
        'car_brand_Toyota': 0, 'car_brand_Volkswagen': 0
    }

    # Set the appropriate columns to 1 based on the input features
    original_format[distance_mapping[input_features['distance'][0]]] = 1
    original_format[fuel_mapping[input_features['fuel'][0]]] = 1
    original_format[seller_mapping[input_features['seller_type'][0]]] = 1
    original_format[owner_mapping[input_features['owner'][0]]] = 1
    original_format[car_brand_mapping[input_features['car_brand'][0]]] = 1

    # Add the rest of the columns
    original_format['mileage_km'] = input_features['mileage_km'][0]
    original_format['engine'] = input_features['engine'][0]
    original_format['seats'] = input_features['seats'][0]
    original_format['car_age'] = input_features['car_age'][0]
    original_format['transmission'] = 1 if input_features['transmission'][0] == 'Automatic' else 0
    original_format['distance_km'] = np.nan  # Placeholder for distance_km, which is not provided in input_features

    return pd.DataFrame([original_format])

# Title
st.title("Car Price Prediction App")
st.write("This app predicts the price of a car based on its features.")

# Load data
df = pd.read_csv('processed_cardekho.csv')
df = transform_dataset(df)

# Display data
if st.checkbox("Show random sample"):
    st.write(df.sample(10))

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

# Convert input_features back to the original format
input_features_original_format = reverse_transform(input_features)

# One-hot encode categorical features (dummy encoding)
input_features_original_format = pd.get_dummies(input_features_original_format)
df_encoded = pd.get_dummies(df.drop(columns=['selling_price']))
input_features_original_format = input_features_original_format.reindex(columns=df_encoded.columns, fill_value=0)

# Load scaler and scale input features
scaler = StandardScaler()
X = df_encoded.drop(columns=['selling_price'])
scaler.fit(X)
input_features_scaled = scaler.transform(input_features_original_format)

# Load model
with open('catboost_model.pkl', 'rb') as
