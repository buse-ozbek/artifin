# Retail Demand Monitoring Pipeline

This project demonstrates a machine learning monitoring pipeline using a retail store inventory dataset.

We simulate incoming production data with realistic drift, log live API predictions, compute monitoring metrics, store them in a database, and visualize everything in Grafana.

---

## Pipeline Overview

1. Reference dataset (baseline)
2. Incoming batch data (with simulated drift) and live API predictions
3. Monitoring (drift + regression performance + prediction distribution)
4. Storage in PostgreSQL (two tables: `metrics` + `prediction_logs`)
5. Visualization in Grafana (two dashboards)

---

## Project Structure
```
.
├── data/
│   ├── reference.csv
│   └── current_batches/
├── scripts/
│   ├── prepare_reference.py
│   ├── generate_batch.py
│   └── calculate_metrics.py
├── docker-compose.yml
└── README.md
```

The training pipeline (`pipeline.py`), the FastAPI service (`app.py`), and the
project-level Dockerfile live in the project root.

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Start Docker

docker compose up -d

### 3. Set environment variables
```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=monitoring
export POSTGRES_USER=retail
export POSTGRES_PASSWORD=retail123
```
### 4. Run pipeline

```bash
python scripts/prepare_reference.py
python scripts/generate_batch.py --size 200
python scripts/calculate_metrics.py
```

Repeat the last two steps multiple times. Use `--size` to vary batch size (e.g. `--size 500`).

### 5. Run the FastAPI service

```bash
docker run -p 8001:8001 \
  -v $(pwd)/../mlruns:/Users/buseozbek/PycharmProjects/PythonProject2/mlruns \
  -e POSTGRES_HOST=host.docker.internal \
  retail-predict-service
```

Send `/predict` requests at http://localhost:8001/docs — each call writes a row to `prediction_logs`.

---

## Adminer

http://localhost:8080

Login:
- System: PostgreSQL
- Server: db
- User: retail
- Password: retail123
- Database: monitoring

Queries:
SELECT * FROM metrics;
SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 50;

---

## Grafana

http://localhost:3000

Login:
- admin / admin

Add PostgreSQL datasource:
- Host: db:5432
- Database: monitoring
- User: retail
- Password: retail123
- TLS/SSL Mode: disable

Two dashboards:
- Retail Monitoring Dashboard — reads `metrics` (drift, MAE, RMSE, R²)
- Retail Prediction Logs Dashboard — reads `prediction_logs` (live API traffic)

---

## Example Queries

MAE over time:
```
SELECT timestamp AS time, mae FROM metrics;
```
Drift share:
```
SELECT timestamp AS time, share_drifted_features FROM metrics;
```
Recent predictions:
```
SELECT timestamp, store_id, category, region, prediction FROM prediction_logs ORDER BY timestamp DESC LIMIT 20;
```

---

## Diagram

![Monitoring Pipeline](monitoring_pipeline.png)
