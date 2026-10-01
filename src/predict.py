import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "xgb_churn_pipeline.joblib"
)


model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """
    Predict churn probability and risk for a single customer.

    Parameters
    ----------
    customer_data : dict
        Customer information using the original Telco feature names.

    Returns
    -------
    dict
        Churn probability, prediction, and risk level.
    """

    customer_df = pd.DataFrame([customer_data])

    probability = model.predict_proba(customer_df)[0, 1]

    prediction = int(probability >= 0.20)

    if probability >= 0.50:
        risk = "High Risk"
    elif probability >= 0.20:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "risk_level": risk
    }