import pandas as pd
import numpy as np

import streamlit as st
import joblib

# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# # Connect to the MongoDB cluster to store the inputs and the prediction
# uri = "**********************************************************"
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['california_housing']  # Create a new database
# collection = db['california_housing_pred'] # Create a new collection/table in the database

def load_model():
    data = joblib.load('xgboost_regressor_knn_regressor_and_scalar.pkl')

    scalar = data['scalar']
    xgboost_regressor_best_params = data['xgb_regressor_best_params']
    knn_regressor_best_params = data['knn_regressor_best_params']

    return scalar, xgboost_regressor_best_params, knn_regressor_best_params

def data_preprocessing(data):
    df = pd.DataFrame([data])

    scalar = load_model()[0]
    df_scaled = scalar.transform(df)

    return df_scaled

def predict_data(data, model):
    data = data_preprocessing(data)

    if model == "XGBoost Regressor":
        xgboost_regressor_best_params = load_model()[1]
        return xgboost_regressor_best_params.predict(data)
    elif model == "KNN Regressor":
        knn_regressor_best_params = load_model()[2]
        return knn_regressor_best_params.predict(data)

def main():
    st.title("California Housing Price Prediction")
    st.write("Enter the following details to predict the price of a house in California")

    MedInc = st.slider("Median income of the people living in the block", 0.0, 15.0, 3.0)
    HouseAge = st.slider("Median house age in the block", 0.0, 50.0, 1.0)
    AveRooms = st.slider("Average number of rooms", 0.0, 5.0, 3.0)
    AveBedrms = st.slider("Average number of bedrooms", 0.0, 3.0, 2.0)
    Population = st.slider("Population of the block", 0.0, 40000.0, 15000.0)
    AveOccup = st.slider("Average house occupancy", 0.0, 10.0, 4.0)
    Latitude = st.slider("Latitude", 32.54, 41.95, 1.0)
    Longitude = st.slider("Longitude", -124.35, -114.31, 1.0)

    model_options = [
        "XGBoost Regressor",
        "KNN Regressor"
    ]

    selected_model = st.selectbox("Select the model", model_options)

    if st.button("Predict the price"):
        user_data = {
            'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude
        }

        if selected_model == "XGBoost Regressor":
            prediction = predict_data(user_data, selected_model)
        elif selected_model == "KNN Regressor":
            prediction = predict_data(user_data, selected_model)

        st.success(f"Prediction using {selected_model}: {float(prediction)} million dollars")

        # user_data["prediction"] = round(float(prediction[0]), 3)    # Add the prediction to the user_data dictionary
        # user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, float) else value for key, value in user_data.items()}    # Convert the values to int or float if they are of type np.integer or np.float
        # collection.insert_one(user_data)    # Insert the user_data dictionary/record to the MongoDB collection

if __name__ == '__main__':
    main()