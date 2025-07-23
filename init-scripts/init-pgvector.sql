-- Initialisation de PostgreSQL avec pgvector pour les embeddings médicaux

-- Activation de l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Table pour stocker les embeddings des documents
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,
    sentence_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384), -- Dimension pour les embeddings simples
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour la recherche par similarité vectorielle (HNSW pour performance)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON document_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index standard pour les requêtes par document_id
CREATE INDEX IF NOT EXISTS idx_document_id 
ON document_embeddings (document_id);

-- Index pour les recherches par sentence_id
CREATE INDEX IF NOT EXISTS idx_sentence_id 
ON document_embeddings (sentence_id);

-- Table pour les métadonnées de documents
CREATE TABLE IF NOT EXISTS document_metadata (
    document_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    file_type VARCHAR(50),
    file_hash VARCHAR(64),
    total_sentences INTEGER,
    specialties TEXT[],
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour les statistiques de recherche
CREATE TABLE IF NOT EXISTS search_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),
    search_type VARCHAR(50),
    results_count INTEGER,
    response_time_ms INTEGER,
    user_specialty VARCHAR(100),
    clicked_results TEXT[],
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour l'analyse des performances
CREATE INDEX IF NOT EXISTS idx_search_analytics_query_hash 
ON search_analytics (query_hash);

CREATE INDEX IF NOT EXISTS idx_search_analytics_timestamp 
ON search_analytics (timestamp);

-- Fonction pour la mise à jour automatique des timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour les mises à jour automatiques
CREATE TRIGGER update_document_embeddings_updated_at 
    BEFORE UPDATE ON document_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_metadata_updated_at 
    BEFORE UPDATE ON document_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insertion de données de test pour validation
INSERT INTO document_embeddings (document_id, sentence_id, content, embedding) VALUES
('test-doc-1', 'test-doc-1_0', 'Le patient présente une douleur thoracique depuis 2 jours.', 
 array_fill(0.1, ARRAY[384])::vector(384))
ON CONFLICT (sentence_id) DO NOTHING;

-- Vérification de l'installation
SELECT 
    'pgvector extension' as component,
    CASE WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') 
         THEN 'INSTALLED' 
         ELSE 'MISSING' 
    END as status
UNION ALL
SELECT 
    'document_embeddings table' as component,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'document_embeddings')
         THEN 'CREATED'
         ELSE 'MISSING'
    END as status;