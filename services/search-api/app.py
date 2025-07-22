# services/search-api/app.py
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
        # OpenSearch
        self.opensearch_client = OpenSearch(
            hosts=[os.getenv('OPENSEARCH_URL', 'http://localhost:9200')],
            use_ssl=False,
            verify_certs=False
        )
        
        # PostgreSQL
        self.postgres_pool = await asyncpg.create_pool(
            os.getenv('POSTGRES_URL'),
            min_size=2,
            max_size=10
        )
        
        # Redis
        self.redis_client = aioredis.from_url(
            os.getenv('REDIS_URL'),
            decode_responses=True
        )
        
        logger.info("Moteur de recherche initialisé")

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Génération d'embedding pour la requête"""
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.embedder.encode([query])[0]
            )
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
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur recherche textuelle: {e}")
            return []

    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Recherche sémantique avec embeddings"""
        try:
            # Génération embedding de la requête
            query_embedding = await self.generate_query_embedding(query)
            
            # Recherche par similarité vectorielle
            async with self.postgres_pool.acquire() as conn:
                query_sql = """
                    SELECT 
                        document_id,
                        sentence_id,
                        content,
                        embedding <-> $1::vector as distance,
                        1 - (embedding <-> $1::vector) as similarity
                    FROM document_embeddings
                    ORDER BY embedding <-> $1::vector
                    LIMIT $2
                """
                
                rows = await conn.fetch(query_sql, query_embedding.tolist(), limit * 2)
                
                # Regroupement par document
                doc_scores = {}
                for row in rows:
                    doc_id = row['document_id']
                    similarity = row['similarity']
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'max_similarity': similarity,
                            'contents': [row['content']],
                        }
                    else:
                        doc_scores[doc_id]['max_similarity'] = max(
                            doc_scores[doc_id]['max_similarity'], 
                            similarity
                        )
                        doc_scores[doc_id]['contents'].append(row['content'])
                
                # Tri et sélection
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
                        'metadata': {}
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Erreur recherche sémantique: {e}")
            return []

    async def hybrid_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """Recherche hybride combinant textuel + sémantique"""
        # Recherches parallèles
        textual_task = self.textual_search(
            search_query.query, 
            search_query.filters, 
            search_query.limit * 2
        )
        semantic_task = self.semantic_search(
            search_query.query, 
            search_query.limit * 2
        )
        
        textual_results, semantic_results = await asyncio.gather(
            textual_task, semantic_task
        )
        
        # Fusion des résultats
        combined_results = {}
        
        # Intégration des résultats textuels
        for result in textual_results:
            doc_id = result['document_id']
            combined_results[doc_id] = {
                'document_id': doc_id,
                'content': result['content'],
                'title': result['title'],
                'textual_score': min(result['score'] / 10.0, 1.0),  # Normalisation
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
        return final_results[:search_query.limit]

# API FastAPI
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
    await search_engine.initialize()

@app.post("/search", response_model=List[Dict])
async def search_documents(search_query: SearchQuery):
    """Recherche intelligente dans les documents médicaux"""
    try:
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
        logger.error(f"Erreur lors de la recherche: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medical-search-api"}

@app.get("/stats")
async def get_search_stats():
    """Statistiques du moteur de recherche"""
    try:
        # Stats PostgreSQL
        async with search_engine.postgres_pool.acquire() as conn:
            doc_count = await conn.fetchval(
                "SELECT COUNT(DISTINCT document_id) FROM document_embeddings"
            )
            embedding_count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_embeddings"
            )
        
        return {
            "indexed_documents": doc_count or 0,
            "unique_documents": doc_count or 0,
            "total_embeddings": embedding_count or 0,
            "index_size": 0
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)