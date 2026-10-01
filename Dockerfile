FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY data/processed/xgb_churn_pipeline.joblib ./data/processed/
COPY data/processed/customer_segmentation.csv ./data/processed/
COPY data/processed/customer_cltv.csv ./data/processed/
COPY data/processed/telco_churn_clean.csv ./data/processed/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]