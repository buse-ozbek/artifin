# Retail Store Inventory — Demand Forecasting

## Project Overview
This project builds a machine learning pipeline to forecast daily product demand across retail stores. The model predicts **Units Sold** based on store context, pricing, weather, promotions, and seasonality.

**Dataset:** [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/) — 73,000+ daily records across multiple stores and products.

**Goal:** Predict how many units of a product will be sold on a given day, enabling smarter inventory management and supply chain optimization.

---

## Project Structure
```
artifin/
├── training_pipeline/
│   ├── pipeline.py            # Train models + MLflow experiment tracking
│   ├── predict_pipeline.py    # Batch predictions from CSV
│   └── app.py                 # FastAPI prediction service
├── model_tracking/            # MLflow experiment tracking folder
├── Dockerfile                 # Container for deployment
├── .gitignore
└── README.md
```

---

## ML Models & Experiments
Three models were trained and compared using MLflow:

| Experiment | Model | Notes |
|---|---|---|
| Exp1 | Linear Regression | Simple baseline |
| Exp2 | Random Forest | 100 trees, ensemble method |
| Exp3 | Gradient Boosting | 200 rounds, learns from mistakes |

**Best model selection:** Voting system across 3 metrics (MAE, RMSE, R²). The model winning the most categories is registered to MLflow Model Registry under the alias `Staging`.

**Results:**
- MAE  ≈ 7.14 (off by ~7 units on average)
- RMSE ≈ 8.36
- R²   ≈ 0.9941 (model explains 99.41% of variance)

---

## Features Used
- Store ID, Product ID, Category, Region
- Inventory Level, Demand Forecast, Price, Discount
- Weather Condition, Holiday/Promotion, Competitor Pricing
- Seasonality, Year, Month, Day of Week

---

## How to Run

### 1. Install dependencies
```bash
pip install mlflow scikit-learn pandas numpy fastapi uvicorn
```

### 2. Start MLflow UI
```bash
mlflow ui --port 5001
# Open http://127.0.0.1:5001
```

### 3. Train models
```bash
python training_pipeline/pipeline.py
```

### 4. Start the API
```bash
uvicorn training_pipeline.app:app --reload --port 8001
# Open http://127.0.0.1:8001/docs
```

### 5. Run batch predictions (optional)
```bash
python training_pipeline/predict_pipeline.py
# Creates predictions.csv
```

---

## API Usage
**Endpoint:** `POST /predict`

**Example request:**
```json
{
  "date": "2024-03-15",
  "store_id": "S001",
  "product_id": "P0005",
  "category": "Electronics",
  "region": "North",
  "inventory_level": 200,
  "demand_forecast": 150.5,
  "price": 49.99,
  "discount": 10,
  "weather_condition": "Sunny",
  "holiday_promotion": 1,
  "competitor_pricing": 52.00,
  "seasonality": "Winter"
}
```

**Example response:**
```json
{
  "prediction": 146
}
```

---

## Docker Deployment
```bash
# Build
docker build -t retail-demand-api .

# Run
docker run -p 8000:8000 retail-demand-api
```

---

## Tech Stack
- **Python 3.9**
- **scikit-learn** — ML models
- **MLflow** — Experiment tracking & model registry
- **FastAPI** — REST API
- **Docker** — Containerization
- **pandas / numpy** — Data processing
