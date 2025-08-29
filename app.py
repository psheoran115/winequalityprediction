import streamlit as st
import numpy as np
import pickle

# Load trained model (make sure wine_model.pkl is in same folder)
model = pickle.load(open("winequality.pkl", "rb"))

# Title
st.title("🍷 Wine Quality Prediction App")

st.write("Enter the wine characteristics below to predict its quality:")

# Collect all 11 features used in training
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, step=1.0)
density = st.number_input("Density", min_value=0.0, max_value=2.0, step=0.0001, format="%.4f")
pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, step=0.1)

# Prediction button
if st.button("Predict Quality"):
    # Arrange inputs in the same order as training
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, pH,
                          sulphates, alcohol]])
    
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
