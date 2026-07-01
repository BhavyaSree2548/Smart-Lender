import os
import pickle
import shutil

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


DATASET_PATH = os.path.join("Dataset", "loan_prediction.csv")
MODEL_PATH = os.path.join("Flask", "rdf.pkl")
SCALER_PATH = os.path.join("Flask", "scale1.pkl")
SCALER_COPY_PATH = os.path.join("Flask", "scale1(1).pkl")


def clean_dataset():
    data = pd.read_csv(DATASET_PATH)
    data = data.drop(columns=["Loan_ID"], errors="ignore")

    data["Dependents"] = data["Dependents"].replace("3+", 3)

    numeric_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].mean())

    for column in categorical_columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    data["Married"] = data["Married"].map({"No": 0, "Yes": 1})
    data["Education"] = data["Education"].map({"Not Graduate": 0, "Graduate": 1})
    data["Self_Employed"] = data["Self_Employed"].map({"No": 0, "Yes": 1})
    data["Property_Area"] = data["Property_Area"].map(
        {"Rural": 0, "Semiurban": 1, "Urban": 2}
    )
    data["Loan_Status"] = data["Loan_Status"].map({"N": 0, "Y": 1})
    data["Dependents"] = pd.to_numeric(data["Dependents"], errors="coerce").fillna(0)

    x = data.drop(columns=["Loan_Status"])
    y = data["Loan_Status"]
    return x, y


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)

    print(f"\n{name}")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_prediction))
    print("Classification Report:")
    print(classification_report(y_test, test_prediction))

    return test_accuracy, model


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    x, y = clean_dataset()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.to_numpy())

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.33, random_state=42, stratify=y
    )

    models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier()),
    ]

    if XGBClassifier is not None:
        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                ),
            )
        )
    else:
        print("\nXGBoost is not installed. Install it with:")
        print("python -m pip install xgboost")
        print("Using Gradient Boosting as a temporary fallback.\n")
        models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=42)))

    best_score = -1
    best_model = None

    for name, model in models:
        score, trained_model = evaluate_model(
            name, model, x_train, x_test, y_train, y_test
        )
        if score > best_score:
            best_score = score
            best_model = trained_model

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(best_model, model_file)

    with open(SCALER_PATH, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    shutil.copyfile(SCALER_PATH, SCALER_COPY_PATH)

    print(f"\nBest testing accuracy: {best_score:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")


if __name__ == "__main__":
    main()
