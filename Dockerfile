# CUDA runtime image matched to the pinned torch (cu121). For GPU training/eval.
# On Colab you do NOT need this image — the notebook installs deps directly.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: serve the API (override CMD to train/eval).
#   train:  docker run ... python scripts/train.py --config configs/qlora_5k.yaml
#   eval:   docker run ... python scripts/run_eval.py --mode base0 --output results/base_0shot
EXPOSE 8000
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
