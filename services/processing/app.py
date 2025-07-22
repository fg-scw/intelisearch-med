import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid
import hashlib
import re

import asyncpg
import aioredis
import aiofiles
from opensearchpy import OpenSearch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from sentence_transformers import SentenceTransformer
import nltk

# Configuration du logging pour un meilleur suivi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Le téléchargement des ressources NLTK ('punkt') est géré dans le Dockerfile.

class DocumentEmbedder:
    """
    Génère des embeddings de haute qualité en utilisant un modèle SentenceTransformer.
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L12-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Modèle d'embedding '{model_name}' chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du chargement du modèle SentenceTransformer: {e}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode une liste de textes en vecteurs d'embeddings."""
        logger.info(f"Génération de {len(texts)} embeddings...")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False
        )
        logger.info(f"✅ Embeddings générés avec succès. Shape: {embeddings.shape}")
        return embeddings

class MedicalDocumentProcessor:
    """
    Orchestre le pipeline de traitement des documents.
    """
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        self.redis_client = None
        model_name = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L12-v2')
        self.embedder = DocumentEmbedder(model_name)
        
    async def initialize(self):
        """Initialise les connexions aux services externes."""
        try:
            self.opensearch_client = OpenSearch(
                hosts=[os.getenv('OPENSEARCH_URL')],
                use_ssl=False, verify_certs=False
            )
            self.postgres_pool = await asyncpg.create_pool(os.getenv('POSTGRES_URL'))
            self.redis_client = aioredis.from_url(os.getenv('REDIS_URL'), decode_responses=True)
            logger.info("✅ Processeur de documents initialisé et connecté aux services.")
        except Exception as e:
            logger.error(f"❌ Échec de l'initialisation des connexions: {e}")
            raise

    async def extract_text_from_file(self, file_path: Path) -> str:
        """Extrait le contenu texte d'un fichier."""
        if file_path.suffix.lower() != '.txt':
            raise ValueError(f"Type de fichier non supporté: {file_path.suffix}")
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()

    def preprocess_medical_text(self, text: str) -> List[str]:
        """Nettoie et segmente le texte en phrases pertinentes."""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = nltk.sent_tokenize(text, language='french')
        return [s.strip() for s in sentences if len(s.split()) > 3]

    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Encapsule l'encodage pour une exécution asynchrone."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embedder.encode, texts)

    async def store_in_opensearch(self, document_id: str, content: str, metadata: Dict):
        """Indexe le document dans OpenSearch."""
        try:
            document_body = { 'content': content, 'title': metadata.get('title', 'Titre inconnu'), **metadata }
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.opensearch_client.index(
                    index='medical_documents',
                    id=document_id,
                    body=document_body,
                    refresh=True
                )
            )
            logger.info(f"📄 Document {document_id} stocké dans OpenSearch.")
        except Exception as e:
            logger.error(f"❌ Erreur lors du stockage OpenSearch pour {document_id}: {e}")
            raise # Propage l'erreur pour qu'elle soit gérée par l'appelant

    async def store_embeddings_in_postgres(self, document_id: str, sentences: List[str], embeddings: np.ndarray):
        """Stocke les embeddings dans PostgreSQL."""
        try:
            async with self.postgres_pool.acquire() as conn:
                data_to_insert = [
                    (
                        str(uuid.uuid4()), 
                        document_id, 
                        f"{document_id}_{i}", 
                        sentence, 
                        # --- CORRECTION APPLIQUÉE ICI ---
                        # Convertir la liste en sa représentation textuelle
                        str(embedding.tolist()) 
                    )
                    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings))
                ]
                
                await conn.executemany("""
                    INSERT INTO document_embeddings (id, document_id, sentence_id, content, embedding, created_at)
                    VALUES ($1, $2, $3, $4, $5::vector, NOW())
                    ON CONFLICT (sentence_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW();
                """, data_to_insert)

            logger.info(f"🧠 {len(embeddings)} embeddings pour le document {document_id} stockés dans PostgreSQL.")
        except Exception as e:
            logger.error(f"❌ Erreur lors du stockage PostgreSQL pour {document_id}: {e}")
            raise # Propage l'erreur

    async def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Pipeline complet pour traiter un fichier."""
        logger.info(f"🚀 Démarrage du traitement pour: {file_path.name}")
        document_id = str(uuid.uuid4())
        try:
            raw_text = await self.extract_text_from_file(file_path)
            if not raw_text.strip():
                raise ValueError("Le document fourni est vide.")

            sentences = self.preprocess_medical_text(raw_text)
            if not sentences:
                raise ValueError("Aucune phrase valide n'a été trouvée.")

            embeddings = await self.generate_embeddings(sentences)
            
            metadata = {
                'title': file_path.name,
                'file_type': file_path.suffix,
                'hash': hashlib.sha256(raw_text.encode()).hexdigest(),
                'sentences_count': len(sentences)
            }
            
            # Le stockage se fait maintenant en série pour mieux gérer les erreurs
            await self.store_in_opensearch(document_id, raw_text, metadata)
            await self.store_embeddings_in_postgres(document_id, sentences, embeddings)
            
            logger.info(f"✅ Traitement de {file_path.name} terminé avec succès (ID: {document_id}).")
            return {
                'document_id': document_id,
                'sentences_processed': len(sentences),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du traitement de {file_path.name} (ID: {document_id}): {e}")
            return {'status': 'error', 'error': str(e), 'filename': file_path.name}

# --- API FastAPI ---
# Remplacement de on_event déprécié par le contexte lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    app.state.processor = MedicalDocumentProcessor()
    await app.state.processor.initialize()
    yield
    # Code exécuté à l'arrêt (si nécessaire)

app = FastAPI(title="Service de Traitement de Documents Médicaux", lifespan=lifespan)

# --- AJOUT DE LA CONFIGURATION CORS ---
# C'est la partie qui résout l'erreur "Failed to fetch"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet à toutes les origines (pour le développement)
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permet tous les en-têtes
)


# Ajout d'un gestionnaire d'exception global pour plus de propreté
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée pour la requête {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Une erreur interne est survenue.", "detail": str(exc)},
    )

@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Endpoint pour uploader et traiter un nouveau document."""
    upload_dir = Path("/app/uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        processor = request.app.state.processor
        result = await processor.process_document(file_path)

        if result['status'] == 'error':
            # Renvoyer une erreur 500 si le traitement interne a échoué
            return JSONResponse(status_code=500, content=result)
        
        return JSONResponse(status_code=201, content=result)
    finally:
        if file_path.exists():
            file_path.unlink()

@app.get("/health")
async def health_check():
    """Endpoint de santé pour la supervision."""
    return {"status": "healthy", "service": "medical-processing"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)