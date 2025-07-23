# import os
# import asyncio
# import logging
# from typing import List, Dict, Any
# from contextlib import asynccontextmanager

# import asyncpg
# from opensearchpy import OpenSearch
# from opensearchpy.exceptions import NotFoundError
# import numpy as np
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import uvicorn
# from sentence_transformers import SentenceTransformer

# # Configuration du logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # --- Classes et Logique Métier ---

# class DocumentEmbedder:
#     def __init__(self, model_name: str):
#         try:
#             self.model = SentenceTransformer(model_name)
#             logger.info(f"✅ Modèle d'embedding '{model_name}' chargé avec succès.")
#         except Exception as e:
#             logger.error(f"❌ Erreur critique lors du chargement du modèle: {e}")
#             raise
#     def encode(self, texts: List[str]) -> np.ndarray:
#         return self.model.encode(texts, show_progress_bar=False)

# class SearchQuery(BaseModel):
#     query: str
#     limit: int = 10
#     search_type: str = "hybrid"
#     rrf_k: int = 60

# class SearchResult(BaseModel):
#     document_id: str
#     title: str
#     snippet: str
#     content: str
#     relevance_score: float = 0.0
#     semantic_score: float = 0.0
#     hybrid_score: float = 0.0

# class MedicalSearchEngine:
#     def __init__(self):
#         self.opensearch_client = None
#         self.postgres_pool = None
#         model_name = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L12-v2')
#         self.embedder = DocumentEmbedder(model_name)
        
#     async def initialize(self):
#         self.opensearch_client = OpenSearch(hosts=[os.getenv('OPENSEARCH_URL')])
#         self.postgres_pool = await asyncpg.create_pool(os.getenv('POSTGRES_URL'))
#         logger.info("✅ Moteur de recherche initialisé.")

#     async def generate_query_embedding(self, query: str) -> np.ndarray:
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(None, lambda: self.embedder.encode([query])[0])

#     async def textual_search(self, query: str, limit: int) -> List[SearchResult]:
#         search_body = {
#             "query": {"multi_match": {"query": query, "fields": ["title^3", "content"], "fuzziness": "AUTO"}},
#             "highlight": {"fields": {"content": {"fragment_size": 150, "number_of_fragments": 2}}},
#             "size": limit
#         }
#         try:
#             response = await asyncio.get_event_loop().run_in_executor(
#                 None, lambda: self.opensearch_client.search(index="medical_documents", body=search_body)
#             )
#             results = []
#             for hit in response['hits']['hits']:
#                 snippet = " ... ".join(hit.get('highlight', {}).get('content', [])) or hit['_source']['content'][:250] + "..."
#                 results.append(SearchResult(
#                     document_id=hit['_id'],
#                     title=hit['_source'].get('title', 'Sans titre'),
#                     snippet=snippet,
#                     content=hit['_source']['content'],
#                     relevance_score=hit['_score']
#                 ))
#             logger.info(f"🔍 Recherche textuelle pour '{query}' a retourné {len(results)} résultats.")
#             return results
#         except Exception as e:
#             logger.error(f"❌ Erreur pendant la recherche textuelle: {e}")
#             return []

#     async def semantic_search(self, query: str, limit: int) -> List[SearchResult]:
#         """
#         --- FONCTION ENTIÈREMENT CORRIGÉE ---
#         Effectue une recherche sémantique puis enrichit les résultats avec les données d'OpenSearch.
#         """
#         try:
#             query_embedding = await self.generate_query_embedding(query)
#             async with self.postgres_pool.acquire() as conn:
#                 rows = await conn.fetch("""
#                     SELECT 
#                         document_id,
#                         content AS best_snippet,
#                         1 - (embedding <=> $1::vector) as similarity
#                     FROM document_embeddings
#                     ORDER BY embedding <=> $1::vector
#                     LIMIT $2
#                 """, str(query_embedding.tolist()), limit * 10)

#             # Agrège par document, en gardant la meilleure similarité et le meilleur snippet
#             doc_scores: Dict[str, Dict[str, Any]] = {}
#             for row in rows:
#                 doc_id = row['document_id']
#                 if doc_id not in doc_scores or row['similarity'] > doc_scores[doc_id]['similarity']:
#                     doc_scores[doc_id] = {'similarity': float(row['similarity']), 'snippet': row['best_snippet']}
            
#             # Récupère les IDs des documents à enrichir
#             doc_ids = list(doc_scores.keys())
#             if not doc_ids:
#                 return []

#             # Enrichit les résultats en récupérant les métadonnées depuis OpenSearch
#             try:
#                 os_response = await asyncio.get_event_loop().run_in_executor(
#                     None, lambda: self.opensearch_client.mget(index="medical_documents", body={"ids": doc_ids})
#                 )
                
#                 enriched_docs = {doc['_id']: doc['_source'] for doc in os_response.get('docs', []) if doc.get('found')}
#             except NotFoundError:
#                 enriched_docs = {} # Gère le cas où aucun document n'est trouvé

#             # Construit les objets SearchResult finaux
#             results = []
#             for doc_id, data in doc_scores.items():
#                 if doc_id in enriched_docs:
#                     source = enriched_docs[doc_id]
#                     results.append(SearchResult(
#                         document_id=doc_id,
#                         title=source.get('title', 'Titre inconnu'),
#                         content=source.get('content', ''),
#                         snippet=data['snippet'], # On garde le snippet le plus pertinent trouvé sémantiquement
#                         semantic_score=data['similarity']
#                     ))
            
#             # Trie les résultats finaux par score de similarité
#             results.sort(key=lambda r: r.semantic_score, reverse=True)
#             logger.info(f"🧠 Recherche sémantique pour '{query}' a retourné {len(results)} résultats enrichis.")
#             return results[:limit]
#         except Exception as e:
#             logger.error(f"❌ Erreur pendant la recherche sémantique: {e}", exc_info=True)
#             return []

#     def reciprocal_rank_fusion(self, search_results: List[List[SearchResult]], k: int) -> List[SearchResult]:
#         fused_scores: Dict[str, float] = {}
#         doc_data: Dict[str, SearchResult] = {} 

#         for results in search_results:
#             for rank, result in enumerate(results):
#                 doc_id = result.document_id
#                 if doc_id not in fused_scores:
#                     fused_scores[doc_id] = 0
#                     doc_data[doc_id] = result
                
#                 doc_data[doc_id].relevance_score = max(doc_data[doc_id].relevance_score, result.relevance_score)
#                 doc_data[doc_id].semantic_score = max(doc_data[doc_id].semantic_score, result.semantic_score)
                
#                 fused_scores[doc_id] += 1 / (k + rank + 1)

#         for doc_id, score in fused_scores.items():
#             doc_data[doc_id].hybrid_score = score
        
#         return sorted(doc_data.values(), key=lambda r: r.hybrid_score, reverse=True)
        
#     async def search(self, query: SearchQuery) -> List[SearchResult]:
#         if query.search_type == "textual":
#             return await self.textual_search(query.query, query.limit)
        
#         if query.search_type == "semantic":
#             return await self.semantic_search(query.query, query.limit)
        
#         logger.info(f"🚀 Lancement de la recherche hybride pour '{query.query}'")
#         textual_task = self.textual_search(query.query, query.limit)
#         semantic_task = self.semantic_search(query.query, query.limit)
        
#         textual_results, semantic_results = await asyncio.gather(textual_task, semantic_task)
        
#         if not textual_results and not semantic_results:
#             return []
        
#         fused_results = self.reciprocal_rank_fusion([textual_results, semantic_results], query.rrf_k)
        
#         return fused_results[:query.limit]

# # --- API FastAPI ---

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.state.search_engine = MedicalSearchEngine()
#     await app.state.search_engine.initialize()
#     yield

# app = FastAPI(title="API de Recherche Médicale", lifespan=lifespan)
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# @app.exception_handler(Exception)
# async def generic_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Erreur non gérée pour la requête {request.url}: {exc}", exc_info=True)
#     return JSONResponse(status_code=500, content={"message": "Une erreur interne est survenue.", "detail": str(exc)})

# ######
# ######
# @app.post("/admin/reset", status_code=200)
# async def reset_databases(request: Request):
#     """
#     Réinitialise complètement les bases de données OpenSearch et PostgreSQL.
#     À utiliser avec précaution.
#     """
#     logger.warning("🚨 Requête de réinitialisation des bases de données reçue.")
#     search_engine: MedicalSearchEngine = request.app.state.search_engine
    
#     # 1. Nettoyer l'index OpenSearch
#     try:
#         if await asyncio.get_event_loop().run_in_executor(None, lambda: search_engine.opensearch_client.indices.exists(index="medical_documents")):
#             await asyncio.get_event_loop().run_in_executor(None, lambda: search_engine.opensearch_client.indices.delete(index="medical_documents"))
#             await asyncio.get_event_loop().run_in_executor(None, lambda: search_engine.opensearch_client.indices.create(index="medical_documents", body={
#                 "mappings": {
#                     "properties": {
#                         "content": {"type": "text"},
#                         "title": {"type": "text"},
#                     }
#                 }
#             }))
#             logger.info("✅ Index OpenSearch 'medical_documents' a été réinitialisé.")
#     except Exception as e:
#         logger.error(f"❌ Erreur lors de la réinitialisation d'OpenSearch: {e}")
#         raise HTTPException(status_code=500, detail=f"Erreur OpenSearch: {e}")

#     # 2. Nettoyer les tables PostgreSQL
#     try:
#         async with search_engine.postgres_pool.acquire() as conn:
#             await conn.execute("TRUNCATE TABLE document_embeddings, document_metadata RESTART IDENTITY;")
#             logger.info("✅ Tables PostgreSQL ont été vidées.")
#     except Exception as e:
#         logger.error(f"❌ Erreur lors de la réinitialisation de PostgreSQL: {e}")
#         raise HTTPException(status_code=500, detail=f"Erreur PostgreSQL: {e}")

#     return {"message": "Toutes les bases de données ont été réinitialisées avec succès."}
# ######
# ######

# @app.post("/search", response_model=List[SearchResult])
# async def perform_search(request: Request, query: SearchQuery):
#     search_engine: MedicalSearchEngine = request.app.state.search_engine
#     results = await search_engine.search(query)
#     return results

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "service": "medical-search-api"}

# @app.get("/stats", response_model=Dict[str, int])
# async def get_stats(request: Request):
#     search_engine: MedicalSearchEngine = request.app.state.search_engine
#     try:
#         async with search_engine.postgres_pool.acquire() as conn:
#             total_embeddings = await conn.fetchval("SELECT COUNT(*) FROM document_embeddings") or 0
#             unique_docs = await conn.fetchval("SELECT COUNT(DISTINCT document_id) FROM document_embeddings") or 0
        
#         os_stats = await asyncio.get_event_loop().run_in_executor(
#             None, lambda: search_engine.opensearch_client.count(index='medical_documents')
#         )
#         indexed_documents = os_stats.get('count', 0)

#         return {
#             "indexed_documents_opensearch": indexed_documents,
#             "unique_documents_postgres": unique_docs,
#             "total_embeddings_postgres": total_embeddings
#         }
#     except Exception as e:
#         logger.error(f"Erreur lors de la récupération des stats: {e}")
#         raise HTTPException(status_code=500, detail="Impossible de récupérer les statistiques.")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import asyncio
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Classes et Logique Métier ---

class DocumentEmbedder:
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Modèle d'embedding '{model_name}' chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du chargement du modèle: {e}")
            raise
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    search_type: str = "hybrid"
    rrf_k: int = 60

class SearchResult(BaseModel):
    document_id: str
    title: str
    snippet: str
    content: str
    relevance_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0

class MedicalSearchEngine:
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        model_name = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L12-v2')
        self.embedder = DocumentEmbedder(model_name)
        
    async def initialize(self):
        self.opensearch_client = OpenSearch(hosts=[os.getenv('OPENSEARCH_URL')])
        self.postgres_pool = await asyncpg.create_pool(os.getenv('POSTGRES_URL'))
        logger.info("✅ Moteur de recherche initialisé.")

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embedder.encode([query])[0])

    async def textual_search(self, query: str, limit: int) -> List[SearchResult]:
        search_body = {
            "query": {"multi_match": {"query": query, "fields": ["title^3", "content"], "fuzziness": "AUTO"}},
            "highlight": {"fields": {"content": {"fragment_size": 150, "number_of_fragments": 2}}},
            "size": limit
        }
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.opensearch_client.search(index="medical_documents", body=search_body)
            )
            results = []
            for hit in response['hits']['hits']:
                snippet = " ... ".join(hit.get('highlight', {}).get('content', [])) or hit['_source']['content'][:250] + "..."
                results.append(SearchResult(
                    document_id=hit['_id'],
                    title=hit['_source'].get('title', 'Sans titre'),
                    snippet=snippet,
                    content=hit['_source']['content'],
                    relevance_score=hit['_score']
                ))
            logger.info(f"🔍 Recherche textuelle pour '{query}' a retourné {len(results)} résultats.")
            return results
        except Exception as e:
            logger.error(f"❌ Erreur pendant la recherche textuelle: {e}")
            return []

    async def semantic_search(self, query: str, limit: int) -> List[SearchResult]:
        try:
            query_embedding = await self.generate_query_embedding(query)
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        document_id,
                        content AS best_snippet,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM document_embeddings
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """, str(query_embedding.tolist()), limit * 10)

            doc_scores: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                doc_id = row['document_id']
                if doc_id not in doc_scores or row['similarity'] > doc_scores[doc_id]['similarity']:
                    doc_scores[doc_id] = {'similarity': float(row['similarity']), 'snippet': row['best_snippet']}
            
            doc_ids = list(doc_scores.keys())
            if not doc_ids:
                return []

            try:
                os_response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.opensearch_client.mget(index="medical_documents", body={"ids": doc_ids})
                )
                enriched_docs = {doc['_id']: doc['_source'] for doc in os_response.get('docs', []) if doc.get('found')}
            except NotFoundError:
                enriched_docs = {}

            results = []
            for doc_id, data in doc_scores.items():
                if doc_id in enriched_docs:
                    source = enriched_docs[doc_id]
                    results.append(SearchResult(
                        document_id=doc_id,
                        title=source.get('title', 'Titre inconnu'),
                        content=source.get('content', ''),
                        snippet=data['snippet'],
                        semantic_score=data['similarity']
                    ))
            
            results.sort(key=lambda r: r.semantic_score, reverse=True)
            logger.info(f"🧠 Recherche sémantique pour '{query}' a retourné {len(results)} résultats enrichis.")
            return results[:limit]
        except Exception as e:
            logger.error(f"❌ Erreur pendant la recherche sémantique: {e}", exc_info=True)
            return []

    def reciprocal_rank_fusion(self, search_results: List[List[SearchResult]], k: int) -> List[SearchResult]:
        fused_scores: Dict[str, float] = {}
        doc_data: Dict[str, SearchResult] = {} 

        for results in search_results:
            for rank, result in enumerate(results):
                doc_id = result.document_id
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_data[doc_id] = result
                
                doc_data[doc_id].relevance_score = max(doc_data[doc_id].relevance_score, result.relevance_score)
                doc_data[doc_id].semantic_score = max(doc_data[doc_id].semantic_score, result.semantic_score)
                
                fused_scores[doc_id] += 1 / (k + rank + 1)

        for doc_id, score in fused_scores.items():
            doc_data[doc_id].hybrid_score = score
        
        return sorted(doc_data.values(), key=lambda r: r.hybrid_score, reverse=True)
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        if query.search_type == "textual":
            return await self.textual_search(query.query, query.limit)
        
        if query.search_type == "semantic":
            return await self.semantic_search(query.query, query.limit)
        
        logger.info(f"🚀 Lancement de la recherche hybride pour '{query.query}'")
        textual_task = self.textual_search(query.query, query.limit)
        semantic_task = self.semantic_search(query.query, query.limit)
        
        textual_results, semantic_results = await asyncio.gather(textual_task, semantic_task)
        
        if not textual_results and not semantic_results:
            return []
        
        fused_results = self.reciprocal_rank_fusion([textual_results, semantic_results], query.rrf_k)
        
        return fused_results[:query.limit]

# --- API FastAPI ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_engine = MedicalSearchEngine()
    await app.state.search_engine.initialize()
    yield

app = FastAPI(title="API de Recherche Médicale", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée pour la requête {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"message": "Une erreur interne est survenue.", "detail": str(exc)})

@app.post("/search", response_model=List[SearchResult])
async def perform_search(request: Request, query: SearchQuery):
    search_engine: MedicalSearchEngine = request.app.state.search_engine
    results = await search_engine.search(query)
    return results

@app.post("/admin/reset", status_code=200, response_model=Dict[str, str])
async def reset_databases(request: Request):
    """
    Réinitialise complètement les bases de données OpenSearch et PostgreSQL.
    À utiliser avec précaution.
    """
    logger.warning("🚨 Requête de réinitialisation des bases de données reçue.")
    search_engine: MedicalSearchEngine = request.app.state.search_engine
    
    # 1. Nettoyer et recréer l'index OpenSearch
    index_name = "medical_documents"
    try:
        loop = asyncio.get_event_loop()
        if await loop.run_in_executor(None, lambda: search_engine.opensearch_client.indices.exists(index=index_name)):
            await loop.run_in_executor(None, lambda: search_engine.opensearch_client.indices.delete(index=index_name))
        
        # Recréer l'index avec le mapping correct
        index_body = {
            "mappings": { "properties": { "content": {"type": "text"}, "title": {"type": "text"} } }
        }
        await loop.run_in_executor(None, lambda: search_engine.opensearch_client.indices.create(index=index_name, body=index_body))
        logger.info(f"✅ Index OpenSearch '{index_name}' a été réinitialisé.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réinitialisation d'OpenSearch: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur OpenSearch: {e}")

    # 2. Vider les tables PostgreSQL
    try:
        async with search_engine.postgres_pool.acquire() as conn:
            # TRUNCATE est plus rapide que DELETE pour vider des tables entières
            await conn.execute("TRUNCATE TABLE document_embeddings, document_metadata RESTART IDENTITY;")
            logger.info("✅ Tables PostgreSQL ont été vidées.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réinitialisation de PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur PostgreSQL: {e}")

    return {"message": "Toutes les bases de données ont été réinitialisées avec succès."}

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    return {"status": "healthy", "service": "medical-search-api"}

@app.get("/stats", response_model=Dict[str, int])
async def get_stats(request: Request):
    search_engine: MedicalSearchEngine = request.app.state.search_engine
    try:
        async with search_engine.postgres_pool.acquire() as conn:
            total_embeddings = await conn.fetchval("SELECT COUNT(*) FROM document_embeddings") or 0
            unique_docs = await conn.fetchval("SELECT COUNT(DISTINCT document_id) FROM document_embeddings") or 0
        
        os_stats = await asyncio.get_event_loop().run_in_executor(
            None, lambda: search_engine.opensearch_client.count(index='medical_documents')
        )
        indexed_documents = os_stats.get('count', 0)

        return {
            "indexed_documents_opensearch": indexed_documents,
            "unique_documents_postgres": unique_docs,
            "total_embeddings_postgres": total_embeddings
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        raise HTTPException(status_code=500, detail="Impossible de récupérer les statistiques.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)