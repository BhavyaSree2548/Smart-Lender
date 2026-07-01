# Smart Lender - Applicant Creditworthiness Prediction

Smart Lender is a machine learning powered Flask web application that predicts whether a loan application is likely to be approved. The project trains and compares Decision Tree, Random Forest, K-Nearest Neighbors and XGBoost classifiers, then saves the best performing model for Flask integration.

## Project Folders

- `Dataset/loan_prediction.csv` - training dataset
- `Training/Loan Prediction using ML.ipynb` - machine learning workflow notebook
- `Flask/app1.py` - Flask application
- `Flask/templates/` - HTML pages
- `Flask/static/css/style.css` - page styling
- `Flask/rdf.pkl` - saved trained model
- `Flask/scale1.pkl` - saved scaler

## Run The Project

```bash
cd E:\SmartLender-Application
python Flask/app1.py
```

Open:

```text
http://127.0.0.1:5000
```

## Re-Train The Model

```bash
python train_model.py
```

This creates:

- `Flask/rdf.pkl`
- `Flask/scale1.pkl`
- `Flask/scale1(1).pkl`

## Install Prerequisites

```bash
python -m pip install -r requirements.txt
```

If `xgboost` is missing, install it separately:

```bash
python -m pip install xgboost
```

If SMOTE is needed in the notebook:

```bash
python -m pip install imbalanced-learn
```

## IBM Cloud Deployment

This project includes `Procfile` and `runtime.txt` for cloud deployment. For IBM Cloud Code Engine or Cloud Foundry, use the same start command:

```bash
gunicorn Flask.app1:app
```
