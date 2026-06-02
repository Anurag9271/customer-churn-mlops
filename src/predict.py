import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"


# Load model
model = joblib.load(MODEL_PATH)


def predict(input_data):

    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(df)[0]

    return prediction


if __name__ == "__main__":

    sample_customer = {
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 1,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 0,
        "OnlineSecurity": 0,
        "OnlineBackup": 1,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Contract": 0,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    result = predict(sample_customer)

    print("Prediction:", result)