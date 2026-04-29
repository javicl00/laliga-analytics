FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo fuente
COPY . .

# Puerto API
EXPOSE 8000

# Entrypoint por defecto: API serving
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
