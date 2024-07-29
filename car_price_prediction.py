# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load and preprocess data
def load_data():
    df = pd.read_csv('cardekho.csv')  # Replace with your dataset path
    return df

# Load the model
def load_model():
    model = xgb.Booster()
    model.load_model('best_model.json')  # Load the model from the JSON file
    return model

# Preprocess input features
def preprocess_input(data, scaler):
    data_scaled = scaler.transform(data)
    return data_scaled

# Main function to run the app
def main():
    st.title("Car Price Prediction App")
    st.write("This app predicts the price of a car based on its features.")
    
    # Load data
    df = load_data()
    
    # Display data
    if st.checkbox("Show raw data"):
        st.write(df.head())
    
    # Define selection arrays
    car_brandTuple = ['Ambassador', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    transmissionTuple = ['Automatic', 'Manual']
    seller_typeTuple = ['Dealer', 'Individual', 'Trustmark Dealer']
    fuelTuple = ['CNG', 'Diesel', 'LPG', 'Petrol']
    ownerTuple = ['First Owner', 'Fourth & Above Owner', 'Second Owner', 'Third Owner']
    
    # User input for new prediction
    st.header("Input Features")
    car_brand = car_brandTuple.index(st.selectbox("Car Brand", car_brandTuple)) + 1
    mileage_km = st.number_input("Mileage (km)", min_value=0, max_value=5000000)
    engine = st.number_input("Engine (cc)", min_value=500, max_value=4000)
    seats = st.number_input("Seats", min_value=2, max_value=8)
    car_age = st.number_input("Car Age", min_value=0, max_value=30)
    transmission = transmissionTuple.index(st.selectbox("Transmission", transmissionTuple)) + 1
    fuel = fuelTuple.index(st.selectbox("Fuel Type", fuelTuple)) + 1
    seller_type = seller_typeTuple.index(st.selectbox("Seller Type", seller_typeTuple))
    owner = ownerTuple.index(st.selectbox("Owner Type", ownerTuple)) + 1
    distance_km = st.number_input("Distance (km)", min_value=0, max_value=500000)
    
    # Prepare input for prediction
    input_features = pd.DataFrame({
        'distance_km': [distance_km],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage_km': [mileage_km],
        'engine': [engine],
        'max_power': [0],  # This field might need to be adjusted based on your data
        'seats': [seats],
        'car_brand': [car_brand],
        'car_age': [car_age]
    })
    
    st.write(input_features)
    
    # Load model
    model = load_model()
    
    if st.button("Predict"):
        # Convert input features to DMatrix
        dmatrix = xgb.DMatrix(input_features)
        
        # Make prediction
        prediction = model.predict(dmatrix)
        
        # Display prediction
        st.subheader("Predicted Selling Price")
        st.write(f"${prediction[0]:,.2f}")
    
main()
