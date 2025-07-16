# Flask application to host our salary regression model in the form of an API

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

model = joblib.load("final_salary_model.pkl")
col_names = joblib.load('salary_column_names.pkl')

@app.route('/')

def hello() -> str:
    return 'Helllo World from flask!!'

@app.route('/predict', methods=['POST'])
def predict():

    # get json request
    user_data = request.json
    # Convert Json request to pandas DataFrame
    df = pd.DataFrame(user_data)
    # Match Column Names
    df = df.reindex(columns=col_names)
    # Get Prediction
    prediction = list(model.predict(df))
    # Return JSON version of Prediciton
    return jsonify({'prediction' : str(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
