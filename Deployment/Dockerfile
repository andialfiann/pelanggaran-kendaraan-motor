FROM python:3.11-slim

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY requirements.txt ./

# (Opsional) install system dependencies jika dibutuhkan
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY Deployment/ .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.headless=true", "--server.enableCORS=false"]
