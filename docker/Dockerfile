# Dockerfile
# Retail Store Inventory — Demand Forecasting API
# Containerizes the FastAPI prediction service

FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all necessary files into the container
COPY training_pipeline/app.py .
COPY training_pipeline/predict_pipeline.py .
COPY models/ models/

# Install all required dependencies
RUN pip install mlflow joblib numpy scikit-learn fastapi uvicorn pandas

# Set MLflow environment variables
# host.docker.internal lets the container talk to your local machine
#ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
#ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001

# Expose port 8000 so the outside world can reach the API
EXPOSE 8000

# Start the FastAPI server when the container runs
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
