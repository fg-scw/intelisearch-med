import os
import asyncio
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
from opensearchpy import OpenSearch
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
    """
    Classe partagée pour charger et utiliser le modèle SentenceTransformer.
    Identique à celle du service de processing pour assurer la cohérence.
    """
    def __init__(self, model_name: str):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Modèle d'embedding '{model_name}' chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du chargement du modèle SentenceTransformer: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes en vecteurs d'embeddings."""
        return self.model.encode(texts, show_progress_bar=False)

class SearchQuery(BaseModel):
    """Modèle Pydantic pour valider les requêtes de recherche."""
    query: str
    limit: int = 10
    search_type: str = "hybrid"
    # Constante pour l'algorithme de fusion RRF. 60 est une valeur par défaut courante.
    rrf_k: int = 60

class SearchResult(BaseModel):
    """Modèle Pydantic pour structurer les résultats de recherche."""
    document_id: str
    title: str
    snippet: str
    content: str
    relevance_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0

class MedicalSearchEngine:
    """
    Moteur de recherche orchestrant les recherches textuelle, sémantique et hybride.
    """
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        model_name = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L12-v2')
        self.embedder = DocumentEmbedder(model_name)
        
    async def initialize(self):
        """Initialise les connexions aux bases de données."""
        self.opensearch_client = OpenSearch(hosts=[os.getenv('OPENSEARCH_URL')])
        self.postgres_pool = await asyncpg.create_pool(os.getenv('POSTGRES_URL'))
        logger.info("✅ Moteur de recherche initialisé et connecté.")

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Génère l'embedding pour la requête de l'utilisateur."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embedder.encode([query])[0])

    async def textual_search(self, query: str, limit: int) -> List[SearchResult]:
        """Effectue une recherche textuelle sur OpenSearch."""
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
                snippet = " ... ".join(hit['highlight']['content']) if 'highlight' in hit else hit['_source']['content'][:250] + "..."
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
        """Effectue une recherche par similarité sémantique sur PostgreSQL avec pgvector."""
        try:
            query_embedding = await self.generate_query_embedding(query)
            async with self.postgres_pool.acquire() as conn:
                # La distance cosinus (opérateur <->) est 1 - similarité. On cherche la plus petite distance.
                rows = await conn.fetch("""
                    SELECT 
                        document_id,
                        content,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM document_embeddings
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """, query_embedding.tolist(), limit * 5) # On récupère plus de résultats pour les agréger

            # Agrège les résultats par document, en ne gardant que le meilleur score
            doc_scores: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                doc_id = row['document_id']
                if doc_id not in doc_scores or row['similarity'] > doc_scores[doc_id]['score']:
                    doc_scores[doc_id] = {'score': row['similarity'], 'content': row['content']}
            
            # Formatte les résultats et trie par score
            results = [
                SearchResult(document_id=doc_id, title=f"Fragment du document {doc_id[:8]}", content=data['content'], snippet=data['content'][:250]+"...", semantic_score=data['score'])
                for doc_id, data in doc_scores.items()
            ]
            results.sort(key=lambda r: r.semantic_score, reverse=True)
            logger.info(f"🧠 Recherche sémantique pour '{query}' a retourné {len(results)} résultats.")
            return results[:limit]
        except Exception as e:
            logger.error(f"❌ Erreur pendant la recherche sémantique: {e}")
            return []

    def reciprocal_rank_fusion(self, search_results: List[List[SearchResult]], k: int) -> List[SearchResult]:
        """Combine plusieurs listes de résultats classés en utilisant l'algorithme RRF."""
        fused_scores: Dict[str, float] = {}
        doc_data: Dict[str, SearchResult] = {} 

        for results in search_results:
            for rank, result in enumerate(results):
                doc_id = result.document_id
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_data[doc_id] = result
                
                # Met à jour les scores pour garder le meilleur de chaque type
                doc_data[doc_id].relevance_score = max(doc_data[doc_id].relevance_score, result.relevance_score)
                doc_data[doc_id].semantic_score = max(doc_data[doc_id].semantic_score, result.semantic_score)
                
                # Ajoute le score RRF pour le classement actuel
                fused_scores[doc_id] += 1 / (k + rank + 1)

        # Assigne le score RRF final et trie les résultats
        for doc_id, score in fused_scores.items():
            doc_data[doc_id].hybrid_score = score
        
        return sorted(doc_data.values(), key=lambda r: r.hybrid_score, reverse=True)
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Point d'entrée principal pour tous les types de recherche."""
        if query.search_type == "textual":
            return await self.textual_search(query.query, query.limit)
        
        if query.search_type == "semantic":
            return await self.semantic_search(query.query, query.limit)
        
        # Logique pour la recherche hybride
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
    """Gère le cycle de vie de l'application (démarrage/arrêt)."""
    app.state.search_engine = MedicalSearchEngine()
    await app.state.search_engine.initialize()
    yield

app = FastAPI(
    title="API de Recherche Médicale",
    description="Fournit une recherche textuelle, sémantique et hybride sur des documents médicaux.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Capture toutes les erreurs non gérées pour une réponse propre."""
    logger.error(f"Erreur non gérée pour la requête {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Une erreur interne est survenue.", "detail": str(exc)},
    )

@app.post("/search", response_model=List[SearchResult])
async def perform_search(request: Request, query: SearchQuery):
    """Endpoint principal pour effectuer une recherche."""
    search_engine: MedicalSearchEngine = request.app.state.search_engine
    try:
        results = await search_engine.search(query)
        return results
    except Exception as e:
        # Bien que le gestionnaire global puisse l'attraper, une gestion locale est plus précise.
        logger.error(f"Erreur dans l'endpoint /search pour la requête '{query.query}': {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur de recherche.")

@app.get("/health")
async def health_check():
    """Endpoint de santé pour la supervision."""
    # Des vérifications plus poussées (ex: ping des DBs) pourraient être ajoutées ici.
    return {"status": "healthy", "service": "medical-search-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)