# addie-models — ADMET prediction service
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    boto3==1.34.14 \
    httpx==0.25.2 \
    numpy==1.24.3 \
    scipy==1.10.1 \
    rdkit==2023.9.1 \
    pydantic==2.5.0 \
    scikit-learn==1.5.2 \
    onnxruntime==1.18.1 \
    azure-storage-blob==12.19.0 \
    huggingface_hub>=0.20.0 \
    catboost>=1.2.0

# Install PyTorch CPU version separately
RUN pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install Chemprop v2 + Lightning for SOTA endpoints (dili, ld50, lipophilicity, pgp_substrate)
# PINNED intentionally. The DILI double-sigmoid handling depends on chemprop v2's
# predict_step already applying the classification sigmoid; an unpinned upgrade could
# silently change that output transform or break .ckpt loading. Bump deliberately and
# re-verify hepatotoxicity parity when changing these. (numpy resolves to 1.26.4.)
RUN pip3 install --no-cache-dir "chemprop==2.2.3" "lightning==2.6.5"

# Install DGL + DGL-Life for GIN supervised masking embeddings (300 dims)
# Required by CatBoost SOTA models trained with 2873-dim features (2573 base + 300 GIN)
# Must match the benchmark: dgllife.model.load_pretrained('gin_supervised_masking')
RUN pip3 install --no-cache-dir dgl>=2.0 dgllife>=0.3.2

# Create working directory
WORKDIR /app

# Copy application code
COPY main.py .
COPY addie_standardized.py .
COPY tdc_sota_integration.py .

# Environment variables
ENV PORT=8025
# Default to the open Hugging Face weights backend (no cloud creds needed).
ENV STORAGE_BACKEND=HF
ENV HF_MODEL_REPO=NovoMCP/addie-models
ENV MODEL_PREFIX=production/
ENV TDC_MODEL_PREFIX=tdc-sota/
ENV TDC_USE_GIN=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8025/health || exit 1

# Run the service
CMD ["python3", "main.py"]
