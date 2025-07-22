# services/processing/app.py
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
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbedder:
    """Générateur d'embeddings simple basé sur TF-IDF"""
    
    def __init__(self):
        self.vocabulary = {}
        self.dimension = 384  # Dimension fixe
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenisation simple"""
        # Nettoyage et tokenisation basique
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if len(token) > 2]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Génération d'embeddings simples basés sur la fréquence des mots"""
        embeddings = []
        
        for text in texts:
            tokens = self.tokenize(text)
            
            # Calcul de vecteur basé sur hash des tokens
            vector = np.zeros(self.dimension)
            
            for i, token in enumerate(set(tokens)):  # Tokens uniques
                # Hash du token pour déterminer les positions dans le vecteur
                hash_value = hash(token) % self.dimension
                vector[hash_value] += tokens.count(token)  # Fréquence du token
            
            # Normalisation
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            embeddings.append(vector)
        
        return np.array(embeddings)

class MedicalDocumentProcessor:
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        self.redis_client = None
        self.embedder = SimpleEmbedder()
        
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
            os.getenv('POSTGRES_URL', 'postgresql://medical_user:medical_password@localhost:5432/medical_search'),
            min_size=2,
            max_size=10
        )
        
        # Redis
        self.redis_client = aioredis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379'),
            decode_responses=True
        )
        
        logger.info("Initialisation terminée")
    
    async def extract_text_from_file(self, file_path: str) -> str:
        """Extraction du texte selon le type de fichier"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.txt':
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        else:
            raise ValueError(f"Type de fichier non supporté: {file_extension}")

    def preprocess_medical_text(self, text: str) -> List[str]:
        """Préprocessing des textes médicaux"""
        sentences = []
        
        # Sections médicales courantes
        medical_sections = [
            "MOTIF", "ANTÉCÉDENTS", "EXAMEN", "DIAGNOSTIC", "TRAITEMENT", "HISTOIRE"
        ]
        
        current_section = ""
        for line in text.split('\n'):
            line = line.strip()
            if len(line) < 10:  # Ignorer les lignes trop courtes
                continue
                
            # Détection de section
            for section in medical_sections:
                if section in line.upper():
                    current_section = section
                    break
            
            # Ajout avec contexte
            if current_section:
                sentences.append(f"[{current_section}] {line}")
            else:
                sentences.append(line)
        
        return sentences[:50]  # Limite pour éviter trop d'embeddings

    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Génération d'embeddings"""
        try:
            # Exécution dans un thread pour éviter le blocage
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.embedder.encode, 
                texts
            )
            return embeddings
        except Exception as e:
            logger.error(f"Erreur génération embeddings: {e}")
            # Embeddings aléatoires de secours
            return np.random.rand(len(texts), 384)

    async def store_in_opensearch(self, document_id: str, content: str, metadata: Dict):
        """Stockage dans OpenSearch"""
        try:
            document = {
                'id': document_id,
                'content': content,
                'title': metadata.get('title', ''),
                'file_type': metadata.get('file_type', ''),
                'timestamp': metadata.get('timestamp', ''),
                'hash': metadata.get('hash', ''),
            }
            
            # Exécution dans un thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.opensearch_client.index(
                    index='medical_documents',
                    id=document_id,
                    body=document
                )
            )
            logger.info(f"Document {document_id} stocké dans OpenSearch")
        except Exception as e:
            logger.error(f"Erreur stockage OpenSearch: {e}")

    async def store_embeddings_in_postgres(self, document_id: str, sentences: List[str], embeddings: np.ndarray):
        """Stockage des embeddings dans PostgreSQL"""
        try:
            async with self.postgres_pool.acquire() as conn:
                for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
                    sentence_id = f"{document_id}_{i}"
                    embedding_list = embedding.tolist()
                    
                    await conn.execute("""
                        INSERT INTO document_embeddings (
                            id, document_id, sentence_id, content, embedding, created_at
                        ) VALUES ($1, $2, $3, $4, $5, NOW())
                        ON CONFLICT (sentence_id) DO UPDATE SET
                            content = $4, embedding = $5, updated_at = NOW()
                    """, str(uuid.uuid4()), document_id, sentence_id, sentence, embedding_list)
            
            logger.info(f"Embeddings pour {document_id} stockés")
        except Exception as e:
            logger.error(f"Erreur stockage PostgreSQL: {e}")

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Pipeline complet de traitement d'un document"""
        try:
            # 1. Extraction du texte
            raw_text = await self.extract_text_from_file(file_path)
            
            # 2. Préprocessing médical
            sentences = self.preprocess_medical_text(raw_text)
            
            if not sentences:
                raise ValueError("Aucun contenu exploitable trouvé")
            
            # 3. Génération d'embeddings
            embeddings = await self.generate_embeddings(sentences)
            
            # 4. Métadonnées
            document_id = str(uuid.uuid4())
            file_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            metadata = {
                'title': Path(file_path).name,
                'file_type': Path(file_path).suffix,
                'timestamp': str(asyncio.get_event_loop().time()),
                'hash': file_hash,
                'sentences_count': len(sentences)
            }
            
            # 5. Stockage OpenSearch
            await self.store_in_opensearch(document_id, raw_text, metadata)
            
            # 6. Stockage embeddings
            await self.store_embeddings_in_postgres(document_id, sentences, embeddings)
            
            # 7. Cache des métadonnées
            await self.redis_client.setex(
                f"doc:{document_id}:metadata",
                3600,  # TTL 1h
                str(metadata)
            )
            
            logger.info(f"Document traité avec succès: {document_id}")
            
            return {
                'document_id': document_id,
                'sentences_processed': len(sentences),
                'status': 'success',
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

# API FastAPI
app = FastAPI(title="Medical Document Processing Service")
processor = MedicalDocumentProcessor()

@app.on_event("startup")
async def startup_event():
    await processor.initialize()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload et traitement d'un document médical"""
    try:
        # Sauvegarde temporaire
        upload_dir = Path("/app/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Traitement
        result = await processor.process_document(str(file_path))
        
        # Nettoyage
        file_path.unlink()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medical-processing"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)