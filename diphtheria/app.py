from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define districts
districts = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", 
    "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai", 
    "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", 
    "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        year = request.form['year']
        district = request.form['district']
        so2 = float(request.form['so2'])
        no2 = float(request.form['no2'])
        pm10 = float(request.form['pm10'])
        pm25 = float(request.form['pm25'])

        # Scale the input data
        input_data = np.array([[so2, no2, pm10, pm25]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction = "Diphtheria Risk: HIGH (1)" if prediction == 1 else "Diphtheria Risk: LOW (0)"
    
    return render_template('index.html', districts=districts, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
