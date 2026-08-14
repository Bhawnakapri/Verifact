# VeriFact — deployable container
# Expects data/processed/{corpus,claims,qrels}.jsonl and models/verifact-biencoder-v1,
# models/verifact-nli-v1 to exist (build them locally in Weeks 1-3, then either
# COPY them in here or mount as a volume — see README's deployment section).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-built artifacts from Weeks 1-3 — swap these COPY lines for a volume mount
# if you'd rather not bake multi-GB model weights into the image.
# COPY data/processed ./data/processed
# COPY models ./models

EXPOSE 8000

ENV VERIFACT_PROCESSED_DIR=data/processed
ENV VERIFACT_DENSE_MODEL=models/verifact-biencoder-v1
ENV VERIFACT_NLI_MODEL=models/verifact-nli-v1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
