#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
Created on Fri Feb 28 13:01:53 2025
©author: emilypinson
"""
# Load libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl
import xlrd

# Define the file name manually
file_name = "AmesHousing.xlsx"  # Set your filename here
st.write(f"Using file: {file_name}")

# Read the dataset from the Excel file
df = pd.read_excel(file_name, sheet_name=0)  # Load first sheet

# Clean column names
df.columns = [col.split('(')[0].strip() for col in df.columns]
df.rename(columns={'SalePrice': 'Sale Price'}, inplace=True)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Handle missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Fill numeric with mean
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])  # Fill categorical with mode

# Encode categorical variables (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split the data into features and target
X = df_encoded.drop(columns=['Sale Price'])
y = df_encoded['Sale Price']

# Train a Multiple Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error (MSE) of the model: {mse}")

# Create the Streamlit web-based app
st.title('Ames Housing Dataset Predictions')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')

def user_input_features():
    Lot_Area = st.sidebar.slider('Lot Area', 1300, 215245, 2000)
    Overall_Qual = st.sidebar.slider('Overall Quality', 1, 10, 5)
    Overall_Cond = st.sidebar.slider('Overall Condition', 1, 10, 5)
    Year_Built = st.sidebar.slider('Year Built', 1872, 2010, 1999)
    Total_Bsmt_SF = st.sidebar.slider('Total Basement SF', 0, 6110, 2500)
    data = {
        'Lot Area': Lot_Area,
        'Overall Qual': Overall_Qual,
        'Overall Cond': Overall_Cond,
        'Year Built': Year_Built,
        'Total Bsmt SF': Total_Bsmt_SF,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Create a full input DataFrame matching X’s structure
full_input_df = pd.DataFrame(columns=X.columns)
full_input_df.loc[0] = 0  # Initialize with zeros

# Fill numeric columns with means from training data
numeric_cols_X = X.select_dtypes(include=['int64', 'float64']).columns
full_input_df[numeric_cols_X] = X[numeric_cols_X].mean()

# Update with user-provided values (ensure column names match)
for col in input_df.columns:
    if col in full_input_df.columns:
        full_input_df[col] = input_df[col]

# Ensure all data is numeric
full_input_df = full_input_df.astype(float)

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the housing price
prediction = model.predict(full_input_df)

# Display the prediction (formatted as currency)
st.subheader('Sale Price Prediction (in dollars)')
st.write(f"${prediction[0]:,.2f}")
