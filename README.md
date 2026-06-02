# Customer Churn Prediction

This project predicts whether a telecom customer will churn using Machine Learning. The project includes data preprocessing, model training, experiment tracking with MLflow, prediction pipeline, FastAPI deployment, and Docker containerization.

## Project Structure

```text
customer-churn-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── main.py
│
├── models/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

Place the dataset file inside:

```text
data/raw/Telco_Cusomer_Churn.csv
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Data Preprocessing

```bash
python src/preprocess.py
```

This creates:

```text
data/processed/churn_processed.csv
```

### 2. Model Training

```bash
python src/train.py
```

Models used:

* Logistic Regression
* Random Forest

This creates:

```text
models/best_model.pkl
```

### 3. Run FastAPI Application

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoint

### POST /predict

Sample Request:

```json
{
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
```

Sample Response:

```json
{
  "prediction": 0
}
```

## MLflow

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

MLflow is used to track:

* Parameters
* Metrics
* Model Experiments

## Docker

Build Docker Image:

```bash
docker build -t churn-api .
```

Run Docker Container:

```bash
docker run -p 8000:8000 churn-api
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* MLflow
* FastAPI
* Docker

## Reproducible Pipeline

Run the complete project using:

```bash
python src/preprocess.py
python src/train.py
uvicorn app.main:app --reload
```

This executes the complete pipeline from data preprocessing to prediction serving.

## Author

Anurag
