import joblib
import pandas as pd


def load_model():
    return joblib.load("models/best_model.pkl")


def predict(model, input_data):

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)

    return prediction[0]


if __name__ == "__main__":

    model = load_model()

    sample_customer = {
        # sample data
    }

    result = predict(model, sample_customer)

    print("Prediction:", result)