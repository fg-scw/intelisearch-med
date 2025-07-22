import asyncio
import asyncpg
import os
import logging
from opensearchpy import AsyncOpenSearch
import aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def wait_for_services():
    """Attendre que les services soient prêts"""
    await asyncio.sleep(15)  # Attendre 15 secondes

async def init_opensearch():
    """Initialisation basique d'OpenSearch"""
    try:
        client = AsyncOpenSearch(
            hosts=[os.getenv('OPENSEARCH_URL', 'http://opensearch:9200')],
            use_ssl=False,
            verify_certs=False
        )
        
        # Créer l'index s'il n'existe pas
        if not await client.indices.exists(index="medical_documents"):
            await client.indices.create(
                index="medical_documents",
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "title": {"type": "text"},
                            "file_type": {"type": "keyword"}
                        }
                    }
                }
            )
            logger.info("Index OpenSearch créé")
        await client.close()
    except Exception as e:
        logger.error(f"Erreur OpenSearch: {e}")

async def init_postgres():
    """Test de connexion PostgreSQL"""
    try:
        conn = await asyncpg.connect(os.getenv('POSTGRES_URL'))
        result = await conn.fetchval("SELECT 1")
        logger.info(f"PostgreSQL connecté: {result}")
        await conn.close()
    except Exception as e:
        logger.error(f"Erreur PostgreSQL: {e}")

async def main():
    logger.info("Initialisation des services...")
    await wait_for_services()
    await asyncio.gather(
        init_opensearch(),
        init_postgres(),
        return_exceptions=True
    )
    logger.info("Initialisation terminée")

if __name__ == "__main__":
    asyncio.run(main())
