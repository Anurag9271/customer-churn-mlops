from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_model, predict

app = FastAPI()

model = load_model()


class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict_churn(data: CustomerData):

    prediction = predict(
        model,
        data.model_dump()
    )

    return {
        "prediction": int(prediction)
    }