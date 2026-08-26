# PIT arena — single container running the always-on arena loop + dashboard.
FROM python:3.12-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pit/ ./pit/

# SQLite DB lives on a mounted volume so it survives restarts/redeploys.
ENV PIT_DB_PATH=/data/pit.db \
    PORT=8080 \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8080

# Starts the arena daemon (background thread) + serves the dashboard.
CMD ["python", "-m", "pit.serve"]
