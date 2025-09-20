# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ---- system packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# ---- application setup ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create runtime user and writable media directory
RUN useradd -ms /bin/bash appuser \
 && mkdir -p /app/media \
 && chown -R appuser:appuser /app
USER appuser

ENV PORT=5050
EXPOSE 5050
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5050", "app:app"]
