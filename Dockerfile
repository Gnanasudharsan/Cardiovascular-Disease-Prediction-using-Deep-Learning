# Dockerfile for Cardiovascular Disease Prediction
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved_models results/figures

# Set permissions
RUN chmod +x main.py

# Expose port for Jupyter notebook (if needed)
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; import catboost; print('Health check passed')" || exit 1

# Default command
CMD ["python", "main.py", "--help"]

# Alternative commands for different use cases:
# For training: CMD ["python", "main.py", "--mode", "train"]
# For prediction: CMD ["python", "main.py", "--mode", "predict"]
# For Jupyter: CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
