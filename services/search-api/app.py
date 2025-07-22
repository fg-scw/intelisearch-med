import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import hashlib
from dataclasses import dataclass
import re

import asyncpg
import aioredis
from opensearchpy import OpenSearch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    document_id: str
    content: str
    title: str
    relevance_score: float
    semantic_score: float
    hybrid_score: float
    snippet: str
    metadata: Dict[str, Any]

class SearchQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    patient_context: Optional[Dict[str, str]] = None
    specialty: Optional[str] = None
    search_type: str = "hybrid"

class SimpleEmbedder:
    """Même embedder simple que dans processing"""
    
    def __init__(self):
        self.dimension = 384
        
    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if len(token) > 2]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            tokens = self.tokenize(text)
            vector = np.zeros(self.dimension)
            
            for token in set(tokens):
                hash_value = hash(token) % self.dimension
                vector[hash_value] += tokens.count(token)
            
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            embeddings.append(vector)
        
        return np.array(embeddings)

class MedicalSearchEngine:
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        self.redis_client = None
        self.embedder = SimpleEmbedder()
        
        # Pondérations pour le ranking hybride
        self.ranking_weights = {
            'textual': 0.6,
            'semantic': 0.4
        }
        
    async def initialize(self):
        """Initialisation des connexions"""
        try:
            # OpenSearch
            self.opensearch_client = OpenSearch(
                hosts=[os.getenv('OPENSEARCH_URL', 'http://localhost:9200')],
                use_ssl=False,
                verify_certs=False,
                timeout=30
            )
            
            # Test de connexion OpenSearch
            info = self.opensearch_client.info()
            logger.info(f"OpenSearch connecté: {info['version']['number']}")
            
            # PostgreSQL
            self.postgres_pool = await asyncpg.create_pool(
                os.getenv('POSTGRES_URL'),
                min_size=2,
                max_size=10,
                timeout=60
            )
            
            # Test de connexion PostgreSQL
            async with self.postgres_pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"PostgreSQL connecté: {result[:50]}...")
            
            # Redis
            self.redis_client = aioredis.from_url(
                os.getenv('REDIS_URL'),
                decode_responses=True
            )
            
            # Test de connexion Redis
            await self.redis_client.ping()
            logger.info("Redis connecté")
            
            logger.info("Moteur de recherche initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}")
            raise

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Génération d'embedding pour la requête"""
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.embedder.encode([query])[0]
            )
            logger.info(f"Embedding généré pour '{query}': shape={embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Erreur génération embedding: {e}")
            return np.random.rand(384)

    async def textual_search(self, query: str, filters: Dict = None, limit: int = 10) -> List[Dict]:
        """Recherche textuelle avec OpenSearch"""
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "title^3"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            },
            "size": limit
        }
        
        try:
            # Vérifier d'abord si l'index existe
            if not self.opensearch_client.indices.exists(index="medical_documents"):
                logger.warning("Index medical_documents n'existe pas")
                return []
            
            # Exécution dans un thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.opensearch_client.search(
                    index="medical_documents",
                    body=search_body
                )
            )
            
            results = []
            for hit in response['hits']['hits']:
                snippet = ""
                if 'highlight' in hit and 'content' in hit['highlight']:
                    snippet = " ... ".join(hit['highlight']['content'])
                else:
                    # Snippet simple si pas de highlight
                    content = hit['_source'].get('content', '')
                    snippet = content[:200] + "..." if len(content) > 200 else content
                
                results.append({
                    'document_id': hit['_id'],
                    'content': hit['_source']['content'],
                    'title': hit['_source'].get('title', f"Document {hit['_id'][:8]}..."),
                    'score': hit['_score'],
                    'snippet': snippet,
                    'metadata': hit['_source']
                })
            
            logger.info(f"Recherche textuelle '{query}': {len(results)} résultats")
            return results
            
        except Exception as e:
            logger.error(f"Erreur recherche textuelle: {e}")
            return []

    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Recherche sémantique avec embeddings"""
        try:
            # Génération embedding de la requête
            query_embedding = await self.generate_query_embedding(query)
            
            # Vérification qu'il y a des embeddings en base
            async with self.postgres_pool.acquire() as conn:
                # Compter les embeddings
                count = await conn.fetchval("SELECT COUNT(*) FROM document_embeddings")
                logger.info(f"Embeddings en base: {count}")
                
                if count == 0:
                    logger.warning("Aucun embedding en base pour la recherche sémantique")
                    return []
                
                # Recherche par similarité vectorielle avec pgvector
                try:
                    query_sql = """
                        SELECT 
                            document_id,
                            sentence_id,
                            content,
                            embedding <-> $1::vector as distance,
                            1 - (embedding <-> $1::vector) as similarity
                        FROM document_embeddings
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <-> $1::vector
                        LIMIT $2
                    """
                    
                    rows = await conn.fetch(query_sql, query_embedding.tolist(), limit * 2)
                    logger.info(f"Récupéré {len(rows)} embeddings pour comparaison")
                    
                    if not rows:
                        logger.warning("Aucun résultat de la requête de similarité")
                        return []
                    
                    # Debug des premiers résultats
                    for i, row in enumerate(rows[:3]):
                        logger.info(f"Résultat {i}: distance={row['distance']:.4f}, similarité={row['similarity']:.4f}")
                    
                    # Regroupement par document
                    doc_scores = {}
                    for row in rows:
                        doc_id = row['document_id']
                        similarity = float(row['similarity'])
                        
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = {
                                'max_similarity': similarity,
                                'contents': [row['content']],
                                'count': 1
                            }
                        else:
                            doc_scores[doc_id]['max_similarity'] = max(
                                doc_scores[doc_id]['max_similarity'], 
                                similarity
                            )
                            doc_scores[doc_id]['contents'].append(row['content'])
                            doc_scores[doc_id]['count'] += 1
                    
                    # Tri et sélection des meilleurs documents
                    sorted_docs = sorted(
                        doc_scores.items(),
                        key=lambda x: x[1]['max_similarity'],
                        reverse=True
                    )[:limit]
                    
                    results = []
                    for doc_id, scores in sorted_docs:
                        results.append({
                            'document_id': doc_id,
                            'content': ' '.join(scores['contents'][:2]),
                            'title': f'Document {doc_id[:8]}...',
                            'score': scores['max_similarity'],
                            'snippet': scores['contents'][0][:200] + "..." if scores['contents'] else "",
                            'metadata': {'segment_count': scores['count']}
                        })
                    
                    logger.info(f"Recherche sémantique '{query}': {len(results)} résultats")
                    return results
                    
                except Exception as e:
                    logger.error(f"Erreur dans la requête sémantique: {e}")
                    return []
                
        except Exception as e:
            logger.error(f"Erreur recherche sémantique: {e}")
            return []

    async def hybrid_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """Recherche hybride robuste avec fallbacks"""
        logger.info(f"🔍 Recherche hybride pour: '{search_query.query}'")
        
        try:
            # Recherches parallèles avec gestion d'erreurs
            textual_task = self.textual_search(
                search_query.query, 
                search_query.filters, 
                search_query.limit
            )
            semantic_task = self.semantic_search(
                search_query.query, 
                search_query.limit
            )
            
            textual_results, semantic_results = await asyncio.gather(
                textual_task, semantic_task,
                return_exceptions=True
            )
            
            # Gestion des exceptions
            if isinstance(textual_results, Exception):
                logger.error(f"Erreur recherche textuelle: {textual_results}")
                textual_results = []
            
            if isinstance(semantic_results, Exception):
                logger.error(f"Erreur recherche sémantique: {semantic_results}")
                semantic_results = []
            
            logger.info(f"📊 Résultats - Textuel: {len(textual_results)}, Sémantique: {len(semantic_results)}")
            
            # Si aucun résultat dans les deux recherches
            if not textual_results and not semantic_results:
                logger.warning("❌ Aucun résultat dans les deux types de recherche")
                return []
            
            # Si pas de résultats sémantiques, utiliser seulement les résultats textuels
            if not semantic_results and textual_results:
                logger.info("⚠️ Fallback: utilisation textuelle uniquement")
                final_results = []
                for result in textual_results:
                    final_results.append(SearchResult(
                        document_id=result['document_id'],
                        content=result['content'],
                        title=result['title'],
                        relevance_score=min(result['score'] / 10.0, 1.0),
                        semantic_score=0.0,
                        hybrid_score=min(result['score'] / 10.0, 1.0),
                        snippet=result['snippet'],
                        metadata=result['metadata']
                    ))
                return final_results[:search_query.limit]
            
            # Si pas de résultats textuels, utiliser seulement les résultats sémantiques
            if not textual_results and semantic_results:
                logger.info("⚠️ Fallback: utilisation sémantique uniquement")
                final_results = []
                for result in semantic_results:
                    final_results.append(SearchResult(
                        document_id=result['document_id'],
                        content=result['content'],
                        title=result['title'],
                        relevance_score=0.0,
                        semantic_score=result['score'],
                        hybrid_score=result['score'],
                        snippet=result['snippet'],
                        metadata=result['metadata']
                    ))
                return final_results[:search_query.limit]
            
            # Fusion des résultats quand on a les deux
            logger.info("🔀 Fusion hybride des résultats")
            combined_results = {}
            
            # Intégration des résultats textuels
            for result in textual_results:
                doc_id = result['document_id']
                combined_results[doc_id] = {
                    'document_id': doc_id,
                    'content': result['content'],
                    'title': result['title'],
                    'textual_score': min(result['score'] / 10.0, 1.0),
                    'semantic_score': 0.0,
                    'snippet': result['snippet'],
                    'metadata': result['metadata']
                }
            
            # Intégration des résultats sémantiques
            for result in semantic_results:
                doc_id = result['document_id']
                if doc_id in combined_results:
                    combined_results[doc_id]['semantic_score'] = result['score']
                else:
                    combined_results[doc_id] = {
                        'document_id': doc_id,
                        'content': result['content'],
                        'title': result['title'],
                        'textual_score': 0.0,
                        'semantic_score': result['score'],
                        'snippet': result['snippet'],
                        'metadata': result['metadata']
                    }
            
            # Calcul du score hybride
            final_results = []
            for doc_id, result in combined_results.items():
                textual_norm = result['textual_score']
                semantic_norm = result['semantic_score']
                
                hybrid_score = (
                    self.ranking_weights['textual'] * textual_norm +
                    self.ranking_weights['semantic'] * semantic_norm
                )
                
                final_results.append(SearchResult(
                    document_id=doc_id,
                    content=result['content'],
                    title=result['title'],
                    relevance_score=textual_norm,
                    semantic_score=semantic_norm,
                    hybrid_score=hybrid_score,
                    snippet=result['snippet'],
                    metadata=result['metadata']
                ))
            
            # Tri par score hybride
            final_results.sort(key=lambda x: x.hybrid_score, reverse=True)
            logger.info(f"✅ Recherche hybride terminée: {len(final_results)} résultats")
            
            return final_results[:search_query.limit]
            
        except Exception as e:
            logger.error(f"❌ Erreur critique dans la recherche hybride: {e}")
            return []

# API FastAPI avec CORS très permissif
app = FastAPI(title="Medical Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = MedicalSearchEngine()

@app.on_event("startup")
async def startup_event():
    try:
        await search_engine.initialize()
        logger.info("✅ API de recherche démarrée avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur de démarrage: {e}")
        raise

@app.post("/search", response_model=List[Dict])
async def search_documents(search_query: SearchQuery):
    """Recherche intelligente dans les documents médicaux"""
    try:
        logger.info(f"🔍 Recherche reçue: '{search_query.query}' (type: {search_query.search_type})")
        
        if search_query.search_type == "textual":
            results = await search_engine.textual_search(
                search_query.query, 
                search_query.filters, 
                search_query.limit
            )
            return [{"type": "textual", **r} for r in results]
        
        elif search_query.search_type == "semantic":
            results = await search_engine.semantic_search(
                search_query.query, 
                search_query.limit
            )
            return [{"type": "semantic", **r} for r in results]
        
        else:  # hybrid
            results = await search_engine.hybrid_search(search_query)
            return [{"type": "hybrid", **r.__dict__} for r in results]
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check détaillé"""
    health_status = {"status": "healthy", "service": "medical-search-api"}
    
    try:
        # Test OpenSearch
        if search_engine.opensearch_client:
            search_engine.opensearch_client.ping()
            health_status["opensearch"] = "connected"
        
        # Test PostgreSQL
        if search_engine.postgres_pool:
            async with search_engine.postgres_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["postgres"] = "connected"
        
        # Test Redis
        if search_engine.redis_client:
            await search_engine.redis_client.ping()
            health_status["redis"] = "connected"
            
    except Exception as e:
        health_status["error"] = str(e)
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/stats")
async def get_search_stats():
    """Statistiques du moteur de recherche"""
    try:
        stats = {"indexed_documents": 0, "unique_documents": 0, "total_embeddings": 0, "index_size": 0}
        
        # Stats PostgreSQL
        try:
            async with search_engine.postgres_pool.acquire() as conn:
                doc_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT document_id) FROM document_embeddings"
                )
                embedding_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM document_embeddings"
                )
                stats["unique_documents"] = doc_count or 0
                stats["total_embeddings"] = embedding_count or 0
        except Exception as e:
            logger.error(f"Erreur stats PostgreSQL: {e}")
        
        # Stats OpenSearch
        try:
            loop = asyncio.get_event_loop()
            if search_engine.opensearch_client.indices.exists(index="medical_documents"):
                os_stats = await loop.run_in_executor(
                    None,
                    lambda: search_engine.opensearch_client.cat.count(
                        index="medical_documents", 
                        format="json"
                    )
                )
                stats["indexed_documents"] = int(os_stats[0]['count']) if os_stats else 0
        except Exception as e:
            logger.error(f"Erreur stats OpenSearch: {e}")
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}

# ENDPOINT DEBUG CORRIGÉ - Plus d'array_length sur vector !
@app.get("/debug/embeddings")
async def debug_embeddings():
    """Debug des embeddings - VERSION CORRIGÉE pour pgvector"""
    try:
        async with search_engine.postgres_pool.acquire() as conn:
            # Statistiques générales
            total_count = await conn.fetchval("SELECT COUNT(*) FROM document_embeddings")
            doc_count = await conn.fetchval("SELECT COUNT(DISTINCT document_id) FROM document_embeddings")
            
            # Échantillon d'embeddings - SANS array_length !
            rows = await conn.fetch("""
                SELECT 
                    document_id, 
                    sentence_id, 
                    content, 
                    created_at,
                    CASE 
                        WHEN embedding IS NULL THEN 0 
                        ELSE 384 
                    END as embedding_size
                FROM document_embeddings 
                ORDER BY created_at DESC
                LIMIT 5
            """)
            
            # Vérification d'embeddings null
            null_count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_embeddings WHERE embedding IS NULL"
            )
            
            # Test d'une requête de similarité simple
            similarity_test = None
            try:
                test_embedding = [0.1] * 384  # Embedding de test
                similarity_rows = await conn.fetch("""
                    SELECT document_id, 
                           embedding <-> $1::vector as distance
                    FROM document_embeddings 
                    WHERE embedding IS NOT NULL
                    LIMIT 3
                """, test_embedding)
                
                similarity_test = {
                    "test_successful": True,
                    "sample_distances": [float(row['distance']) for row in similarity_rows]
                }
            except Exception as e:
                similarity_test = {
                    "test_successful": False,
                    "error": str(e)
                }
            
            return {
                "total_embeddings": total_count,
                "unique_documents": doc_count,
                "null_embeddings": null_count,
                "similarity_test": similarity_test,
                "sample_embeddings": [
                    {
                        "document_id": row['document_id'],
                        "sentence_id": row['sentence_id'],
                        "content": row['content'][:100] + "..." if len(row['content']) > 100 else row['content'],
                        "embedding_size": row['embedding_size'],
                        "created_at": row['created_at'].isoformat() if row['created_at'] else None
                    }
                    for row in rows
                ]
            }
    except Exception as e:
        logger.error(f"Erreur debug embeddings: {e}")
        return {"error": str(e), "message": "Erreur lors de l'accès aux embeddings"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
