.PHONY: setup notebook serve docker-build docker-run clean

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "\nSetup complete. Activate with: source venv/bin/activate"

# Launch Jupyter Notebook
notebook:
	jupyter notebook notebooks/

# Start FastAPI server
serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t churn-prediction-api .

docker-run:
	docker run -p 8000:8000 churn-prediction-api

# View MLflow UI
mlflow-ui:
	mlflow ui --port 5000

# Clean generated files
clean:
	rm -rf mlruns/ models/*.joblib outputs/ __pycache__/ src/__pycache__/ .ipynb_checkpoints/
