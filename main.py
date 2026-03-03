import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("retail_store_inventory.csv")
print(df.columns.tolist())

# Create target: 1 = High demand, 0 = Low demand
median_units = df["Units Sold"].median()
df["High Demand"] = (df["Units Sold"] > median_units).astype(int)

# Select features
features = ["Price", "Discount", "Inventory Level", "Units Ordered", "Competitor Pricing"]
X = df[features]
y = df["High Demand"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLflow experiment
mlflow.set_experiment("retail_store_experiment")

with mlflow.start_run(run_name="logistic_regression_v1"):
    model = LogisticRegression(max_iter=5000, solver = "saga")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("features", features)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")