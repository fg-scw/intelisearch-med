# services/processing/app.py
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid
import hashlib

import asyncpg
import aioredis
import aiofiles
from opensearchpy import AsyncOpenSearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Configuration des modèles médicaux
MEDICAL_MODELS = {
    'clinical_bert': 'emilyalsentzer/Bio_ClinicalBERT',
    'biobert': 'dmis-lab/biobert-v1.1',
    'sentence_clinical': 'sentence-transformers/all-MiniLM-L12-v2'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDocumentProcessor:
    def __init__(self):
        self.opensearch_client = None
        self.postgres_pool = None
        self.redis_client = None
        self.medical_model = None
        self.sentence_model = None
        
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
            os.getenv('POSTGRES_URL', 'postgresql://medical_user:medical_password@localhost:5432/medical_search'),
            min_size=2,
            max_size=10
        )
        
        # Redis
        self.redis_client = aioredis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379'),
            decode_responses=True
        )
        
        # Chargement des modèles
        await self.load_models()
        logger.info("Initialisation terminée")
    
    async def load_models(self):
        """Chargement des modèles de transformation"""
        try:
            # Modèle sentence-transformers pour embeddings généraux
            self.sentence_model = SentenceTransformer(
                os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L12-v2')
            )
            
            # Modèle médical spécialisé
            medical_model_name = os.getenv('MEDICAL_MODEL_NAME', 'emilyalsentzer/Bio_ClinicalBERT')
            self.medical_tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
            self.medical_model = AutoModel.from_pretrained(medical_model_name)
            
            logger.info("Modèles chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            raise

    async def extract_text_from_file(self, file_path: str) -> str:
        """Extraction du texte selon le type de fichier"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.txt':
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        
        elif file_extension == '.pdf':
            # Import PyPDF2 ou pdfplumber pour PDF
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        
        else:
            raise ValueError(f"Type de fichier non supporté: {file_extension}")

    def preprocess_medical_text(self, text: str) -> List[str]:
        """Préprocessing spécifique aux textes médicaux"""
        # Segmentation intelligente (phrases, sections cliniques)
        sentences = []
        
        # Division par sections médicales courantes
        medical_sections = [
            "MOTIF DE CONSULTATION", "ANTÉCÉDENTS", "EXAMEN CLINIQUE",
            "EXAMENS COMPLÉMENTAIRES", "DIAGNOSTIC", "TRAITEMENT",
            "HISTOIRE DE LA MALADIE", "EXAMEN PHYSIQUE"
        ]
        
        current_section = ""
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Détection de section
            for section in medical_sections:
                if section in line.upper():
                    current_section = section
                    break
            
            # Ajout avec contexte de section
            if len(line) > 20:  # Filtrer les lignes trop courtes
                sentences.append(f"[{current_section}] {line}" if current_section else line)
        
        return sentences

    async def generate_medical_embeddings(self, texts: List[str]) -> np.ndarray:
        """Génération d'embeddings avec modèle médical spécialisé"""
        embeddings = []
        
        for text in texts:
            # Tokenisation
            inputs = self.medical_tokenizer(
                text, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            # Génération d'embeddings
            with torch.no_grad():
                outputs = self.medical_model(**inputs)
                # Moyenne des tokens pour obtenir l'embedding de la phrase
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)

    async def store_in_opensearch(self, document_id: str, content: str, metadata: Dict):
        """Stockage dans OpenSearch pour recherche textuelle"""
        document = {
            'id': document_id,
            'content': content,
            'title': metadata.get('title', ''),
            'file_type': metadata.get('file_type', ''),
            'timestamp': metadata.get('timestamp', ''),
            'hash': metadata.get('hash', ''),
            'sections': metadata.get('sections', [])
        }
        
        await self.opensearch_client.index(
            index='medical_documents',
            id=document_id,
            body=document
        )

    async def store_embeddings_in_postgres(self, document_id: str, sentences: List[str], embeddings: np.ndarray):
        """Stockage des embeddings dans PostgreSQL avec pgvector"""
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
            embeddings = await self.generate_medical_embeddings(sentences)
            
            # 4. Métadonnées
            document_id = str(uuid.uuid4())
            file_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            metadata = {
                'title': Path(file_path).name,
                'file_type': Path(file_path).suffix,
                'timestamp': asyncio.get_event_loop().time(),
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