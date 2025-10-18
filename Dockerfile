# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install "dvc[s3]" mlflow dvclive pytest

# Copy project files
COPY dvc.yaml .
COPY params.yaml .
COPY .dvc/ .dvc/
COPY src/ src/
COPY model/ model/
COPY app.py .              # <--- Add this
COPY template/ template/   # if you have templates

# Expose port if you want to serve via FastAPI or Flask
EXPOSE 8000

# Fetch DVC-tracked data from S3
RUN dvc pull --force || echo "Some data missing, proceeding..."

# Default command: run the DVC pipeline
#CMD ["dvc", "repro"]
CMD ["python", "app.py"]