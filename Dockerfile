FROM python:3.11-slim

# Install system deps for opencv headless (fixes libGL if any)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY requirements.txt .
COPY setup.sh .

# Run setup script to force numpy downgrade
RUN chmod +x setup.sh && ./setup.sh

# Copy rest of app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "admin_app_embeddings.py", "--server.port=8501", "--server.address=0.0.0.0"]