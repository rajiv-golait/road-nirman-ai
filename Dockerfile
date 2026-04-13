FROM python:3.12-slim

# Install system dependencies required by OpenCV and ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory (used if loading local .pt files later)
RUN mkdir -p models

# Render / Railway / local: platform sets PORT at runtime
EXPOSE 8000
ENV PORT=8000

# Shell form so ${PORT} expands (Render assigns PORT automatically)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
