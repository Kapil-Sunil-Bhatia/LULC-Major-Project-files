from flask import Flask, request, render_template
import numpy as np
import pickle
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load metadata and model
with open('metadata.pkl', 'rb') as file:
    metadata = pickle.load(file)

model = load_model(metadata['model_path'])

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        features = [
            float(request.form[field])
            for field in [
                'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
                'PhysicalActivity', 'DietQuality', 'SleepQuality', 'PollutionExposure',
                'PollenExposure', 'DustExposure', 'PetAllergy', 'FamilyHistoryAsthma',
                'HistoryOfAllergies', 'Eczema', 'HayFever', 'GastroesophagealReflux',
                'LungFunctionFEV1', 'LungFunctionFVC', 'Wheezing', 'ShortnessOfBreath',
                'ChestTightness', 'Coughing', 'NighttimeSymptoms', 'ExerciseInduced'
            ]
        ]
    except ValueError:
        return "Invalid input. Please enter valid numeric values for all fields."

    # Prepare data for prediction
    input_data = np.array([features])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_text = "Asthma Positive" if prediction[0][0] > 0.5 else "Asthma Negative"

    return render_template('result.html', prediction=prediction_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
