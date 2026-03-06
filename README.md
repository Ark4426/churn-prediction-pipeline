# Customer Churn Prediction — End-to-End ML Pipeline

An end-to-end machine learning pipeline for predicting telecom customer churn, from data exploration to model serving via REST API. Built to demonstrate production-ready ML engineering practices.

## Architecture

```
notebooks/
  ├── 01_EDA.ipynb           ← Exploratory data analysis + visualizations
  └── 02_Training.ipynb      ← Model training with sklearn Pipeline + MLflow
src/
  └── serve.py               ← FastAPI prediction endpoint
data/                        ← Raw dataset (Kaggle)
models/                      ← Serialized model artifacts (.joblib)
Dockerfile                   ← Containerized deployment
```

## Key Features

- **Jupyter Notebooks**: Interactive EDA and model training with rich visualizations
- **Scikit-learn Pipeline**: Preprocessing (imputation, scaling, encoding) and model chained into a single reproducible object
- **Feature Engineering**: Derived features — average monthly spend, tenure groups, service count
- **MLflow Tracking**: All experiments logged with hyperparameters, metrics, and model artifacts
- **Model Comparison**: Train and benchmark Logistic Regression, Random Forest, and Gradient Boosting
- **REST API**: FastAPI endpoint with single and batch prediction, auto-generated docs at `/docs`
- **Docker**: Fully containerized for reproducible deployment

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Ark4426/churn-prediction-pipeline.git
cd churn-prediction-pipeline

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle and place it in `data/`:

```bash
# Option A: Kaggle CLI
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip

# Option B: Manual download from Kaggle → save as data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run Notebooks

```bash
jupyter notebook
```

**Notebook 1 — EDA** (`notebooks/01_EDA.ipynb`):
- Churn distribution and class imbalance analysis
- Numeric feature distributions by churn status
- Churn rates across contract type, internet service, payment method
- Correlation heatmap and scatter plot analysis
- Service adoption vs churn patterns
- Summary of key insights for feature engineering

**Notebook 2 — Training** (`notebooks/02_Training.ipynb`):
- Feature engineering (AvgMonthlySpend, TenureGroup, NumServices)
- scikit-learn Pipeline with ColumnTransformer
- Train 3 models: Logistic Regression, Random Forest, Gradient Boosting
- MLflow experiment tracking for all runs
- Model comparison with visualizations
- Confusion matrix, ROC curve, feature importance
- Export best model for FastAPI serving

### 4. View Experiment Results (MLflow)

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Serve Predictions (FastAPI)

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

**API Docs**: http://localhost:8000/docs

**Test with curl:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 211.05
  }'
```

**Response:**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8234,
  "risk_level": "HIGH"
}
```

### 6. Docker Deployment

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Gradient Boosting | ~0.80 | ~0.65 | ~0.55 | ~0.59 | ~0.85 |
| Random Forest | ~0.79 | ~0.63 | ~0.52 | ~0.57 | ~0.83 |
| Logistic Regression | ~0.75 | ~0.55 | ~0.72 | ~0.62 | ~0.83 |

*Note: Results will vary slightly. Logistic Regression with `class_weight="balanced"` trades accuracy for recall — useful when missing churners is more costly than false alarms.*

## Tech Stack

- **Python 3.11**
- **Jupyter Notebook** — Interactive EDA and model development
- **scikit-learn** — Pipeline, ColumnTransformer, model training
- **MLflow** — Experiment tracking, model registry
- **FastAPI** — REST API for model serving
- **Docker** — Containerization
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization

## Project Structure Rationale

This project is structured as a real-world ML pipeline would be in production:

1. **Data exploration** (`01_EDA.ipynb`) — Understand the data before modeling
2. **Training** (`02_Training.ipynb`) — Reproducible pipeline with experiment tracking
3. **Serving** (`serve.py`) — Model exposed as a microservice
4. **Containerization** (`Dockerfile`) — Environment-agnostic deployment

The scikit-learn `Pipeline` ensures preprocessing is always applied consistently between training and inference — a common source of bugs in production ML systems.

## License

MIT
