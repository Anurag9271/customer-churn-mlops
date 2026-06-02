import pandas as pd
from sklearn.preprocessing import LabelEncoder


RAW_PATH = "data/raw/Telco_Cusomer_Churn.csv"
PROCESSED_PATH = "data/processed/churn_processed.csv"


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):

    # Remove customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    # Encode categorical columns
    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":

    df = load_data(RAW_PATH)

    df = preprocess_data(df)

    save_data(df, PROCESSED_PATH)

    print("Preprocessing completed successfully.")