import os
import pickle

import numpy as np
from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rdf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scale1.pkl")

app = Flask(__name__)


def load_pickle(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception:
        return None


model = load_pickle(MODEL_PATH)
scale = load_pickle(SCALER_PATH)


def fallback_predict(data):
    income = data[5] + data[6]
    loan_amount = data[7]
    credit_history = data[9]

    score = 0
    score += 45 if credit_history == 1 else -45
    score += 20 if income >= 6000 else 10 if income >= 3500 else -10
    score += 15 if loan_amount <= 150 else 5 if loan_amount <= 250 else -20
    score += 10 if data[3] == 1 else -5
    score += 8 if data[10] == 1 else 3 if data[10] == 2 else 0

    return 1 if score >= 30 else 0


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    return render_template("input.html")


@app.route("/submit", methods=["POST"])
def submit():
    gender = 1 if request.form["gender"] == "Female" else 0
    married = 1 if request.form["married"] == "Yes" else 0
    dependents = request.form["dependents"]
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 1 if request.form["education"] == "Graduate" else 0
    self_employed = 1 if request.form["self_employed"] == "Yes" else 0
    applicant_income = float(request.form["applicant_income"])
    coapplicant_income = float(request.form["coapplicant_income"])
    loan_amount = float(request.form["loan_amount"])
    loan_amount_term = float(request.form["loan_amount_term"])
    credit_history = float(request.form["credit_history"])
    property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[
        request.form["property_area"]
    ]

    data = [
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area,
    ]

    input_data = np.array([data])

    if model is not None and scale is not None:
        input_data = scale.transform(input_data)
        prediction = model.predict(input_data)[0]
    elif model is not None:
        prediction = model.predict(input_data)[0]
    else:
        prediction = fallback_predict(data)

    if int(prediction) == 0:
        result = "Loan will not be Approved"
    else:
        result = "Loan will be Approved"

    return render_template("output.html", result=result, prediction=int(prediction))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
