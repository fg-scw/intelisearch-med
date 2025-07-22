# services/init/init.py
import asyncio
import asyncpg
import aioredis
from opensearchpy import AsyncOpenSearch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_opensearch():
    """Initialisation des index OpenSearch"""
    client = AsyncOpenSearch(
        hosts=[os.getenv('OPENSEARCH_URL')],
        use_ssl=False,
        verify_certs=False
    )
    
    # Configuration de l'index pour documents médicaux
    index_config = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "medical_analyzer"
                },
                "title": {
                    "type": "text",
                    "analyzer": "medical_analyzer",
                    "boost": 2.0
                },
                "file_type": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "hash": {"type": "keyword"},
                "sections": {"type": "keyword"}
            }
        },
        "settings": {
            "analysis": {
                "analyzer": {
                    "medical_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "medical_synonyms",
                            "medical_stop_words"
                        ]
                    }
                },
                "filter": {
                    "medical_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "MI,myocardial infarction,heart attack",
                            "CHF,congestive heart failure,heart failure",
                            "HTN,hypertension,high blood pressure",
                            "DM,diabetes mellitus,diabetes"
                        ]
                    },
                    "medical_stop_words": {
                        "type": "stop",
                        "stopwords": ["patient", "presents", "history"]
                    }
                }
            }
        }
    }
    
    try:
        # Suppression de l'index s'il existe
        await client.indices.delete(index="medical_documents", ignore=[404])
        
        # Création du nouvel index
        await client.indices.create(index="medical_documents", body=index_config)
        logger.info("Index OpenSearch créé avec succès")
        
        # Template pour les logs de recherche
        log_template = {
            "index_patterns": ["search_logs_*"],
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "results_count": {"type": "integer"},
                    "response_time": {"type": "integer"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        await client.indices.put_template(name="search_logs", body=log_template)
        logger.info("Template de logs créé")
        
    except Exception as e:
        logger.error(f"Erreur initialisation OpenSearch: {e}")
    finally:
        await client.close()

async def init_postgres():
    """Vérification de la configuration PostgreSQL"""
    try:
        conn = await asyncpg.connect(os.getenv('POSTGRES_URL'))
        
        # Vérification de pgvector
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        
        if result:
            logger.info("Extension pgvector disponible")
            
            # Test de connectivité avec la table embeddings
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_embeddings"
            )
            logger.info(f"Table embeddings: {count} entrées")
        else:
            logger.error("Extension pgvector manquante")
            
        await conn.close()
        
    except Exception as e:
        logger.error(f"Erreur initialisation PostgreSQL: {e}")

async def init_redis():
    """Initialisation du cache Redis"""
    try:
        redis = aioredis.from_url(os.getenv('REDIS_URL'))
        
        # Test de connectivité
        await redis.ping()
        logger.info("Redis connecté")
        
        # Configuration du cache
        await redis.config_set('maxmemory-policy', 'allkeys-lru')
        
        await redis.close()
        
    except Exception as e:
        logger.error(f"Erreur initialisation Redis: {e}")

async def main():
    """Initialisation complète des services"""
    logger.info("Début de l'initialisation des services")
    
    # Attente des services
    await asyncio.sleep(10)
    
    # Initialisation parallèle
    await asyncio.gather(
        init_opensearch(),
        init_postgres(), 
        init_redis()
    )
    
    logger.info("Initialisation terminée")

if __name__ == "__main__":
    asyncio.run(main())
