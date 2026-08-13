# Backend image, sized for Hugging Face Spaces' free CPU tier.
#
# Two deliberate choices:
#   1. The embedding model is baked in at build time. Free Spaces have
#      ephemeral disk, so a model downloaded at runtime would be re-fetched on
#      every cold start — ~90MB and 30s the user waits for, every time.
#   2. The sample corpus is ingested at build time too, so a restarted Space
#      always has something to answer questions about instead of greeting
#      visitors with an empty library.
FROM python:3.11-slim

# HF Spaces runs containers as uid 1000 and expects the app on 7860.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    ANCHOR_DB_PATH=/home/user/data/anchor.db

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user:user backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER user
COPY --chown=user:user backend/ ./
COPY --chown=user:user data/samples /home/user/data/samples

# Warm the ONNX model into the image layer, then build the index.
RUN python -c "from fastembed import TextEmbedding; \
        TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')" \
 && python scripts/ingest_cli.py /home/user/data/samples

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
