from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.datasets import load_iris
import streamlit as st

def main():
    st.title('Hello, Streamlit!')
    st.write('This is a simple Streamlit app.')
app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the Iris dataset to get target names
iris = load_iris()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert data to numpy array
    input_data = np.array(data['features']).reshape(1, -1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the prediction
    return jsonify({'prediction': int(prediction[0]), 'class': iris.target_names[prediction[0]]})


if __name__ == '__main__':
    app.run(debug=True)
