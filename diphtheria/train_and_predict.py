import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, ExpSineSquared, DotProduct, WhiteKernel
)
from sklearn.model_selection import train_test_split
import joblib

# 1. Generate synthetic dataset with continuous case counts
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'PM2.5': np.random.uniform(0, 300, n_samples),
    'PM10' : np.random.uniform(0, 400, n_samples),
    'NO2'  : np.random.uniform(0, 200, n_samples),
    'SO2'  : np.random.uniform(0, 100, n_samples),
    'CO'   : np.random.uniform(0, 10,  n_samples),
    'O3'   : np.random.uniform(0, 200, n_samples),
    # simulate random day‐of‐year
    'day_of_year': np.random.randint(1, 366, n_samples)
})

# create a “risk_score” in [0,1] via weighted pollutant fractions
weights = np.array([0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
max_vals = np.array([300, 400, 200, 100, 10, 200])
fractions = df[['PM2.5','PM10','NO2','SO2','CO','O3']].values / max_vals
risk_score = fractions.dot(weights)

# draw Diphtheria_Cases from a Poisson(risk_score * scale); ensures continuous-like counts
scale_factor = 10
df['Diphtheria_Cases'] = np.random.poisson(lam=risk_score * scale_factor)

# save for inspection
df.to_csv('diptheria_air_quality_continuous_synth.csv', index=False)
print("Synthetic continuous dataset saved to diptheria_air_quality_continuous_synth.csv")

# 2. Prepare feature matrix X and log‐transformed target y_log
FEATURES = ['PM2.5','PM10','NO2','SO2','CO','O3','day_of_year']
X = df[FEATURES].values
y = df['Diphtheria_Cases'].values
y_clipped = np.clip(y, 0, None)
y_log = np.log1p(y_clipped)

# 3. Train/test split (we’ll train on all for a final model, but you can split if you like)
# X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
# For demonstration, train on full set:
X_train, y_train = X, y_log

# 4. Define composite GPR kernel and fit
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=50, length_scale_bounds=(1e-2,1e2)) +
    ConstantKernel(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=50, periodicity=365, periodicity_bounds=(1,500)) +
    ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3,1e3)) +
    WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5,1e1))
)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
gpr.fit(X_train, y_train)
print("GPR training complete.")
print("Learned kernel:", gpr.kernel_)

# 5. Save the trained GPR model
MODEL_PATH = 'diphtheria_gpr_model.joblib'
joblib.dump(gpr, MODEL_PATH)
print(f"Trained GPR model saved to {MODEL_PATH}")

# 6. Helper: predict case count + yes/no risk
def predict_diphtheria_risk_gpr(pm25, pm10, no2, so2, co, o3, day_of_year=None):
    """
    Returns (predicted_cases, risk_flag) where risk_flag is 'Yes' if cases>0.
    If day_of_year is None, uses today's day-of-year.
    """
    if day_of_year is None:
        day_of_year = datetime.now().timetuple().tm_yday

    x_new = np.array([[pm25, pm10, no2, so2, co, o3, day_of_year]])
    pred_log, std = gpr.predict(x_new, return_std=True)
    pred_cases = np.expm1(pred_log)[0]
    return pred_cases, ('Yes' if pred_cases > 4 else 'No')

# 7. Example predictions
examples = [
    {'pm25': 150, 'pm10':200, 'no2':80, 'so2':30, 'co':2, 'o3':100},
    {'pm25':  20, 'pm10': 10, 'no2':10, 'so2': 5, 'co':0,'o3': 10},
]

print("\nExample Predictions (GPR):")
for ex in examples:
    cases, risk = predict_diphtheria_risk_gpr(**ex)
    print(f"  Input: {ex} → Predicted Cases: {cases:.2f}, Risk: {risk}")
