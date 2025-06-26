import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the pre-trained model and encoders
model = tf.keras.models.load_model('model.h5')

with open('ohe_geo.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app title
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox("Geography", options=ohe_geo.categories_[0])
gender = st. selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=1000000.0)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=30, value=5)
number_of_products = st.selectbox("Number of Products", options=[1, 2, 3, 4, 5, 6])
has_credit_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# OHE for Geography
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Combine all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write("Prediction Probability: {:.2f}".format(prediction_proba))

if prediction_proba > 0.5:
    st.write("The customer is likely to churn with a probability of {:.2f}%".format(prediction_proba * 100))
else:
    st.write("The customer is likely to stay with a probability of {:.2f}%".format((1 - prediction_proba) * 100))