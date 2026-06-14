FROM python:3.11-slim

WORKDIR /app

# hdbscan compiles a C extension at pip install time — gcc/g++ are required
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download models at Docker build time so the first live request isn't slow.
# Without this, the first /analyze call would trigger a ~520 MB download mid-request
# and guaranteed timeout on Render's end.
RUN python -c "from transformers import pipeline; from sentence_transformers import SentenceTransformer; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); SentenceTransformer('all-MiniLM-L6-v2'); print('Models cached successfully.')"

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
