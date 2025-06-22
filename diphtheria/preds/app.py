from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, ExpSineSquared, DotProduct, WhiteKernel
)
import joblib
import os

# --- Load or train GPR model ---
MODEL_PATH = 'diphtheria_gpr_model.joblib'
if os.path.exists(MODEL_PATH):
    gpr = joblib.load(MODEL_PATH)
else:
    # Generate or load synthetic continuous dataset
    if os.path.exists('diptheria_air_quality_continuous_synth.csv'):
        df = pd.read_csv('diptheria_air_quality_continuous_synth.csv')
    else:
        # Fallback: generate synthetic
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'PM2.5': np.random.uniform(0,300,n_samples),
            'PM10':  np.random.uniform(0,400,n_samples),
            'NO2':   np.random.uniform(0,200,n_samples),
            'SO2':   np.random.uniform(0,100,n_samples),
            'CO':    np.random.uniform(0,10,n_samples),
            'O3':    np.random.uniform(0,200,n_samples),
            'day_of_year': np.random.randint(1,366,n_samples)
        })
        weights = np.array([0.3,0.25,0.2,0.1,0.1,0.05])
        max_vals = np.array([300,400,200,100,10,200])
        fractions = df[['PM2.5','PM10','NO2','SO2','CO','O3']].values / max_vals
        score = fractions.dot(weights)
        df['Diphtheria_Cases'] = np.random.poisson(lam=score * 10)
        df.to_csv('diptheria_air_quality_continuous_synth.csv', index=False)
    
    FEATURES = ['PM2.5','PM10','NO2','SO2','CO','O3','day_of_year']
    X = df[FEATURES].values
    y = df['Diphtheria_Cases'].values
    y_log = np.log1p(np.clip(y, 0, None))

    kernel = (
        ConstantKernel(1.0,(1e-3,1e3)) * RBF(length_scale=50,length_scale_bounds=(1e-2,1e2)) +
        ConstantKernel(1.0,(1e-3,1e3)) * ExpSineSquared(length_scale=50, periodicity=365, periodicity_bounds=(1,500)) +
        ConstantKernel(1.0,(1e-3,1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3,1e3)) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5,1e1))
    )
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gpr.fit(X, y_log)
    joblib.dump(gpr, MODEL_PATH)

# --- Flask app ---
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    predicted_cases = None
    suggestions = []
    if request.method == 'POST':
        # Pollutant inputs
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        no2  = float(request.form['NO2'])
        so2  = float(request.form['SO2'])
        co   = float(request.form['CO'])
        o3   = float(request.form['O3'])
        # current day-of-year
        doy = datetime.now().timetuple().tm_yday

        x_new = np.array([[pm25, pm10, no2, so2, co, o3, doy]])
        pred_log, std = gpr.predict(x_new, return_std=True)
        predicted_cases = float(np.expm1(pred_log)[0])
        result = 'Yes' if predicted_cases > 4 else 'No'

        # Diphtheria advice
        if result == 'Yes':
            suggestions.append('Consult a healthcare professional for testing and vaccination.')
            suggestions.append('Start antibiotic therapy (e.g., erythromycin) and receive antitoxin if diagnosed.')
        else:
            suggestions.append('Low predicted risk; maintain routine vaccinations.')

        # Air quality tips
        suggestions.append('Monitor AQI and limit outdoor exposure when pollution is high.')
        suggestions.append('Use air purifiers indoors or wear N95 masks if needed.')
        suggestions.append('Ensure proper ventilation; avoid indoor smoking or open waste burning.')

    return render_template(
        'index.html',
        result=result,
        predicted_cases=predicted_cases,
        suggestions=suggestions
    )

if __name__ == '__main__':
    app.run(debug=True)
