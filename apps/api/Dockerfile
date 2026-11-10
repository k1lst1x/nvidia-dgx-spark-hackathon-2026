FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Depth Anything V2 source is Apache-2.0. Metric weights are intentionally
# supplied separately through /models rather than downloaded during a build.
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git /opt/depth-anything-v2
COPY *.py ./
COPY botsort.yaml .
ENV PORT=8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
