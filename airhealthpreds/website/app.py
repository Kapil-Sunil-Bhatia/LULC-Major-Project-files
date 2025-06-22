from flask import Flask, render_template, request
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Feature names (should match your training data columns)
feature_names = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]

# Class names (modify based on your model's output classes)
class_names = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]

# Load representative data for LIME (replace with your actual training data)
X_train = np.random.rand(100, len(feature_names)) * 100  # Example synthetic data for LIME
lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification"
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect user inputs from the form
        user_inputs = [float(request.form[feature]) for feature in feature_names]
        input_array = np.array(user_inputs).reshape(1, -1)

        # Model prediction
        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]

        # Generate LIME explanation
        explanation = lime_explainer.explain_instance(
            data_row=input_array[0],
            predict_fn=model.predict_proba
        )
        explanation_html = explanation.as_html()

        # Render results on the same page
        return render_template(
            "index.html",
            features=feature_names,
            predicted_class=class_names[prediction],
            probabilities=probabilities,
            explanation_html=explanation_html
        )

    # Render the form for GET request
    return render_template("index.html", features=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
