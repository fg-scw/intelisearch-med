# services/search-api/app.py
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import hashlib
from dataclasses import dataclass

import asyncpg
import aioredis
from opensearchpy import AsyncOpenSearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query
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
    search_type: str = "hybrid"  # textual, semantic, hybrid

class MedicalSearchEngine:
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        self.redis_client = None
        self.medical_model = None
        self.medical_tokenizer = None
        
        # Pondérations pour le ranking hybride
        self.ranking_weights = {
            'textual': 0.4,
            'semantic': 0.5,
            'context': 0.1
        }
        
        # Spécialités médicales et leurs pondérations
        self.specialty_boosts = {
            'cardiologie': {'heart', 'cardiac', 'cardiovascular', 'ecg'},
            'pneumologie': {'lung', 'respiratory', 'pneumonia', 'asthma'},
            'neurologie': {'brain', 'neurological', 'seizure', 'stroke'},
            'oncologie': {'cancer', 'tumor', 'oncology', 'chemotherapy'}
        }
        
    async def initialize(self):
        """Initialisation des connexions et modèles"""
        # OpenSearch
        self.opensearch_client = AsyncOpenSearch(
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
        
        # Modèles
        await self.load_models()
        logger.info("Moteur de recherche initialisé")
    
    async def load_models(self):
        """Chargement des modèles de recherche sémantique"""
        try:
            # Modèle médical pour embeddings de requête
            medical_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
            self.medical_tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
            self.medical_model = AutoModel.from_pretrained(medical_model_name)
            
            logger.info("Modèles de recherche chargés")
        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            raise

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Génération d'embedding pour la requête"""
        inputs = self.medical_tokenizer(
            query,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.medical_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding

    async def textual_search(self, query: str, filters: Dict = None, limit: int = 10) -> List[Dict]:
        """Recherche textuelle avec OpenSearch"""
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "title^3"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
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
        
        # Ajout des filtres
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})
            
            if filter_clauses:
                search_body["query"]["bool"]["filter"] = filter_clauses
        
        try:
            response = await self.opensearch_client.search(
                index="medical_documents",
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                snippet = ""
                if 'highlight' in hit and 'content' in hit['highlight']:
                    snippet = " ... ".join(hit['highlight']['content'])
                
                results.append({
                    'document_id': hit['_id'],
                    'content': hit['_source']['content'],
                    'title': hit['_source'].get('title', ''),
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
                
                # Regroupement par document avec score agrégé
                doc_scores = {}
                for row in rows:
                    doc_id = row['document_id']
                    similarity = row['similarity']
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'max_similarity': similarity,
                            'avg_similarity': similarity,
                            'contents': [row['content']],
                            'count': 1
                        }
                    else:
                        doc_scores[doc_id]['max_similarity'] = max(
                            doc_scores[doc_id]['max_similarity'], 
                            similarity
                        )
                        doc_scores[doc_id]['avg_similarity'] = (
                            (doc_scores[doc_id]['avg_similarity'] * doc_scores[doc_id]['count'] + similarity) 
                            / (doc_scores[doc_id]['count'] + 1)
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
                    # Récupération des métadonnées du document
                    doc_metadata = await self.get_document_metadata(doc_id)
                    
                    results.append({
                        'document_id': doc_id,
                        'content': ' '.join(scores['contents'][:3]),  # Top 3 sentences
                        'title': doc_metadata.get('title', ''),
                        'score': scores['max_similarity'],
                        'avg_score': scores['avg_similarity'],
                        'snippet': scores['contents'][0][:200] + "..." if scores['contents'] else "",
                        'metadata': doc_metadata
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Erreur recherche sémantique: {e}")
            return []

    async def get_document_metadata(self, document_id: str) -> Dict:
        """Récupération des métadonnées d'un document"""
        # Tentative de récupération depuis le cache
        cached = await self.redis_client.get(f"doc:{document_id}:metadata")
        if cached:
            try:
                return eval(cached)  # Note: En production, utiliser json.loads
            except:
                pass
        
        # Récupération depuis OpenSearch
        try:
            response = await self.opensearch_client.get(
                index="medical_documents",
                id=document_id
            )
            return response['_source']
        except:
            return {}

    def calculate_context_boost(self, result: Dict, patient_context: Dict = None, specialty: str = None) -> float:
        """Calcul du boost contextuel"""
        boost = 1.0
        
        # Boost par spécialité
        if specialty and specialty in self.specialty_boosts:
            specialty_terms = self.specialty_boosts[specialty]
            content = result.get('content', '').lower()
            
            for term in specialty_terms:
                if term in content:
                    boost += 0.1
        
        # Boost par contexte patient
        if patient_context:
            age = patient_context.get('age')
            if age:
                if 'pediatric' in result.get('content', '').lower() and int(age) < 18:
                    boost += 0.15
                elif 'geriatric' in result.get('content', '').lower() and int(age) > 65:
                    boost += 0.15
        
        return min(boost, 1.5)  # Limite le boost à 150%

    async def hybrid_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """Recherche hybride combinant textuel + sémantique + contexte"""
        # Vérification du cache
        cache_key = f"search:{hashlib.md5(search_query.query.encode()).hexdigest()}"
        cached_results = await self.redis_client.get(cache_key)
        
        if cached_results and not search_query.patient_context:
            logger.info("Résultats depuis le cache")
            return json.loads(cached_results)
        
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
        
        # Fusion et scoring hybride
        combined_results = {}
        
        # Intégration des résultats textuels
        for result in textual_results:
            doc_id = result['document_id']
            combined_results[doc_id] = {
                'document_id': doc_id,
                'content': result['content'],
                'title': result['title'],
                'textual_score': result['score'],
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
        
        # Calcul du score hybride avec boost contextuel
        final_results = []
        for doc_id, result in combined_results.items():
            # Normalisation des scores
            textual_norm = result['textual_score'] / 10.0 if result['textual_score'] > 0 else 0
            semantic_norm = result['semantic_score']
            
            # Score hybride pondéré
            hybrid_score = (
                self.ranking_weights['textual'] * textual_norm +
                self.ranking_weights['semantic'] * semantic_norm
            )
            
            # Boost contextuel
            context_boost = self.calculate_context_boost(
                result, 
                search_query.patient_context, 
                search_query.specialty
            )
            
            final_score = hybrid_score * context_boost
            
            final_results.append(SearchResult(
                document_id=doc_id,
                content=result['content'],
                title=result['title'],
                relevance_score=textual_norm,
                semantic_score=semantic_norm,
                hybrid_score=final_score,
                snippet=result['snippet'],
                metadata=result['metadata']
            ))
        
        # Tri par score hybride
        final_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        final_results = final_results[:search_query.limit]
        
        # Mise en cache (sans contexte patient pour éviter les fuites)
        if not search_query.patient_context:
            await self.redis_client.setex(
                cache_key, 
                300,  # 5 minutes
                json.dumps([r.__dict__ for r in final_results], default=str)
            )
        
        return final_results

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
        # Stats OpenSearch
        os_stats = await search_engine.opensearch_client.indices.stats(
            index="medical_documents"
        )
        
        # Stats PostgreSQL
        async with search_engine.postgres_pool.acquire() as conn:
            doc_count = await conn.fetchval(
                "SELECT COUNT(DISTINCT document_id) FROM document_embeddings"
            )
            embedding_count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_embeddings"
            )
        
        return {
            "indexed_documents": os_stats['_all']['total']['docs']['count'],
            "unique_documents": doc_count,
            "total_embeddings": embedding_count,
            "index_size": os_stats['_all']['total']['store']['size_in_bytes']
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)