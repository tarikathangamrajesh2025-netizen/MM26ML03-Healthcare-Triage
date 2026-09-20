from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("model/triage_model.pkl")

print("ML model loaded successfully!")
print("Model classes:", model.classes_)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    patient_data = pd.DataFrame([data])

    prediction = model.predict(patient_data)[0]

    probabilities = model.predict_proba(patient_data)[0]

    probability_dict = {
        model.classes_[i]: float(probabilities[i])
        for i in range(len(model.classes_))
    }

    return jsonify({
        "prediction": prediction,
        "probabilities": probability_dict
    })


if __name__ == "__main__":
    app.run(debug=True)