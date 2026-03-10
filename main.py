import mlflow
import mlflow.sklearn
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    return df

# Step 2: Preprocess data
def preprocess_data(df):
    median_units = df["Units Sold"].median()
    df["High Demand"] = (df["Units Sold"] > median_units).astype(int)
    features = ["Price", "Discount", "Inventory Level", "Units Ordered", "Demand Forecast", "Competitor Pricing"]
    X = df[features]
    y = df["High Demand"]
    return X, y, features

# Step 3: Train model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=5000, solver="saga")
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return accuracy, f1

# Main pipeline
def run_pipeline():
    mlflow.set_experiment("retail_store_experiment")

    with mlflow.start_run(run_name="logistic_regression_pipeline"):
        # Load
        df = load_data()
        print("Data loaded")

        # Preprocess
        X, y, features = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Data preprocessed")

        # Train
        model = train_model(X_train, y_train)
        print("Model trained")

        # Evaluate
        accuracy, f1 = evaluate_model(model, X_test, y_test)
        print("Model evaluated")

        # Log to MLflow
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features", features)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"\n Accuracy: {accuracy:.2f}")
        print(f" F1 Score: {f1:.2f}")

run_pipeline()
