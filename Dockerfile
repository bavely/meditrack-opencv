# ---- base image ----
FROM python:3.11-slim

# System deps for OpenCV video + codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy and install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py ./app.py

# Non-root (optional)
RUN useradd -ms /bin/bash appuser

# Create media dir for outputs
RUN mkdir -p /app/media && chown appuser:appuser /app/media

USER appuser

ENV PORT=5050
EXPOSE 5050

# Gunicorn server
# - 2 workers, 1 thread each is a good start for CPU-bound OpenCV tasks; tune as needed.
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5050", "app:app"]
