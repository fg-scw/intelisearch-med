FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système et Python
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install asyncpg opensearch-py aioredis

# Script d'initialisation
COPY app.py .

CMD ["python", "app.py"]