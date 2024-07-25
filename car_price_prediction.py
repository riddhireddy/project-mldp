import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

# Function to map and print unique values
def map_and_print_unique(df, column):
    if column not in df.columns:
        st.write(f"Error: Column '{column}' not found in dataframe.")
        st.write("Columns present in dataframe:", df.columns.tolist())
        raise KeyError(f"Column '{column}' not found in dataframe.")
    
    unique_values = sorted(df[column].unique())
    mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
    df[column] = df[column].map(mapping)
    st.write(f"Unique values for {column}: {unique_values}")
    return df, mapping

# Load and preprocess data
@st.cache
def load_data():
    df = pd.read_csv('cardekho.csv')  # Replace with your dataset path
    st.write("Columns in the loaded dataframe:", df.columns.tolist())
    return df

# Transform dataset function
def transform_dataset(df):
    st.write("Columns before transformation:", df.columns.tolist())
    
    # Feature engineering
    df.rename(columns={'mileage(km/ltr/kg)': 'mileage_km', 'km_driven': 'distance_km'}, inplace=True)
    
    if 'name' not in df.columns:
        st.write("Error: 'name' column not found in dataframe.")
        raise KeyError("'name' column not found in dataframe.")
    
    df['car_brand'] = df['name'].apply(lambda x: x.split()[0])
    df['car_age'] = 2024 - df['year']
    df.drop(columns=['year', 'name'], inplace=True)
    
    st.write("Columns after renaming and feature engineering:", df.columns.tolist())

    df, fuel_mapping = map_and_print_unique(df, 'fuel')
    df, seller_mapping = map_and_print_unique(df, 'seller_type')
    df, owner_mapping = map_and_print_unique(df, 'owner')
    df, car_brand_mapping = map_and_print_unique(df, 'car_brand')
    df['transmission'] = df['transmission'].apply(lambda x: 1 if x == 'Manual' else 0)
    
    return df, fuel_mapping, seller_mapping, owner_mapping, car_brand_mapping

# Reverse transform input features
def reverse_transform(input_features, mappings):
    fuel_mapping, seller_mapping, owner_mapping, car_brand_mapping = mappings
    
    # Initialize the original format dictionary
    original_format = {
        'fuel': fuel_mapping[input_features['fuel'][0]],
        'seller_type': seller_mapping[input_features['seller_type'][0]],
        'owner': owner_mapping[input_features['owner'][0]],
        'car_brand': car_brand_mapping[input_features['car_brand'][0]],
        'transmission': 1 if input_features['transmission'][0] == 'Manual' else 0,
        'mileage_km': input_features['mileage_km'][0],
        'engine': input_features['engine'][0],
        'seats': input_features['seats'][0],
        'car_age': input_features['car_age'][0],
        'distance_km': input_features['distance_km'][0],
        'max_power': input_features['max_power'][0]
    }
    
    return pd.DataFrame([original_format])

def main():
    # Title
    st.title("Car Price Prediction App")
    st.write("This app predicts the price of a car based on its features.")

    # Load data
    df = load_data()
    st.write("Initial dataframe loaded. Here are the first few rows:")
    st.write(df.head())

    df, fuel_mapping, seller_mapping, owner_mapping, car_brand_mapping = transform_dataset(df)

    # Display data
    if st.checkbox("Show random sample"):
        st.write(df.sample(10))

    # Create selection arrays
    car_brandArr = sorted(car_brand_mapping.keys())
    transmissionArr = ['Manual', 'Automatic']
    fuelArr = sorted(fuel_mapping.keys())
    seller_typeArr = sorted(seller_mapping.keys())
    ownerArr = sorted(owner_mapping.keys())
    distanceArr = df['distance_km'].unique()

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
    distance_km = st.sidebar.number_input("Distance (km)", min_value=0, max_value=500000)
    max_power = st.sidebar.number_input("Max Power (bhp)", min_value=0, max_value=300)

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
        'distance_km': [distance_km],
        'max_power': [max_power]
    })

    # Convert input_features back to the original format
    mappings = (fuel_mapping, seller_mapping, owner_mapping, car_brand_mapping)
    input_features_original_format = reverse_transform(input_features, mappings)

    # Verify that all columns are numeric
    st.write("Input features before scaling:")
    st.write(input_features_original_format)
    
    # Check for any non-numeric columns or missing values
    if not all(input_features_original_format.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        st.write("Error: Non-numeric columns found in input features.")
        raise ValueError("Non-numeric columns found in input features.")
    
    if input_features_original_format.isnull().any().any():
        st.write("Error: Missing values found in input features.")
        raise ValueError("Missing values found in input features.")

    # Load scaler and scale input features
    scaler = StandardScaler()
    scaler.fit(df.drop(columns=['selling_price']))
    input_features_scaled = scaler.transform(input_features_original_format)

    # Load model
    with open('xgboost_best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make prediction
    prediction = model.predict(input_features_scaled)

    # Display prediction
    st.subheader("Predicted Selling Price")
    st.write(f"${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
