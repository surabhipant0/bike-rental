from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature columns
model = pickle.load(open("bike_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # One-hot encode categorical features
    categorical = ['season','weathersit','mnth','weekday','workingday','yr','holiday']
    input_df = pd.get_dummies(input_df, columns=categorical, drop_first=True)

    # Ensure same feature columns as training
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Scale numerical features
    numerical = ['temp','atemp','hum','windspeed']
    input_df[numerical] = scaler.transform(input_df[numerical])

    # Predict
    prediction = model.predict(input_df)[0]

    return jsonify({"prediction": round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
