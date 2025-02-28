# This app is for educaiton demonstration purpose that teaches students how to develop and deploy an interactive web based engineering application app.
# Data source: uc irvine machine learning repository 

# Load libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl
import xlrd

# Load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
data = pd.read_excel(url, sheet_name='Sheet1')

# Clean column names
data.columns = [col.split('(')[0].strip() for col in data.columns]
data.rename(columns={'Concrete compressive strength': 'Strength'}, inplace=True)

# Assuming no missing values, split the data into features and target
X = data.drop(columns=['Strength'])
y = data['Strength']

# Train a Multiple Regression Model 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Create the streamlit web-based app

# Title of the app
st.title('Concrete Compressive Strength Prediction')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')

def user_input_features():
    Cement = st.sidebar.slider('Cement', 0, 540, 100)
    Blast_Furnace_Slag = st.sidebar.slider('Blast Furnace Slag', 0, 359, 0)
    Fly_Ash = st.sidebar.slider('Fly Ash ', 0, 200, 0)
    Water = st.sidebar.slider('Water', 0, 228, 100)
    Superplasticizer = st.sidebar.slider('Superplasticizer', 0, 32, 0)
    Coarse_Aggregate = st.sidebar.slider('Coarse Aggregate', 800, 1145, 1000)
    Fine_Aggregate = st.sidebar.slider('Fine Aggregate', 594, 992, 800)
    Age = st.sidebar.slider('Age', 1, 365, 28)
    
    data = {
        'Cement': Cement,
        'Blast Furnace Slag': Blast_Furnace_Slag,
        'Fly Ash': Fly_Ash,
        'Water': Water,
        'Superplasticizer': Superplasticizer,
        'Coarse Aggregate': Coarse_Aggregate,
        'Fine Aggregate': Fine_Aggregate,
        'Age': Age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the compressive strength
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Predicted Concrete Compressive Strength (MPa)')
st.write(prediction[0])
