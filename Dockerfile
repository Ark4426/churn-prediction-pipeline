FROM python:3.11-slim

WORKDIR /app

# Install only production dependencies (no jupyter needed)
COPY requirements.txt .
RUN pip install --no-cache-dir pandas scikit-learn fastapi uvicorn pydantic joblib numpy

# Copy serving code and trained model
COPY src/ src/
COPY models/ models/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
