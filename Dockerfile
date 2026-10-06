FROM python:3.12.9-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/run-001/best.pt INFERENCE_THREADS=2
WORKDIR /app
COPY requirements.txt constraints.txt ./
# CPU wheels avoid CUDA libraries in a CPU inference deployment.
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -c constraints.txt \
    && useradd --uid 10001 --create-home detector
COPY src/ src/
COPY app/ app/
# Prepare and evaluate the real model before building this image.
COPY artifacts/run-001/ artifacts/run-001/
USER detector
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
