
# services/init/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances
RUN pip install asyncpg opensearch-py aioredis

# Script d'initialisation
COPY init.py .

CMD ["python", "init.py"]
