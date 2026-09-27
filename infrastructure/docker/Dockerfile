# Development image. GPU serving needs an image with CUDA-compatible PyTorch.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY fomo ./fomo
COPY training ./training
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir --no-deps .
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8765
CMD ["python", "-m", "fomo.api.server"]