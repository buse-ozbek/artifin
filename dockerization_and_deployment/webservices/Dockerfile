# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY training_pipeline/app.py .
RUN pip install "mlflow==3.1.4" "numpy==2.0.2" "pandas==2.3.3" "scikit-learn==1.5.2" joblib fastapi uvicorn pydantic "psycopg[binary]"
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5001
EXPOSE 8001
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]