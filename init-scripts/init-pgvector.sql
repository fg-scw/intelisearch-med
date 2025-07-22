-- init-scripts/init-pgvector.sql
-- Initialisation de PostgreSQL avec pgvector pour les embeddings médicaux

-- Activation de l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Table pour stocker les embeddings des documents
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,
    sentence_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768), -- Dimension pour Bio_ClinicalBERT
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

-- Table pour les métadonnées de documents (cache)
CREATE TABLE IF NOT EXISTS document_metadata (
    document_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    file_type VARCHAR(50),
    file_hash VARCHAR(64),
    total_sentences INTEGER,
    specialties TEXT[], -- Array de spécialités détectées
    keywords TEXT[], -- Mots-clés extraits
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour les statistiques de recherche (analytics)
CREATE TABLE IF NOT EXISTS search_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),
    search_type VARCHAR(50), -- textual, semantic, hybrid
    results_count INTEGER,
    response_time_ms INTEGER,
    user_specialty VARCHAR(100),
    clicked_results TEXT[], -- IDs des documents cliqués
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour l'analyse des performances
CREATE INDEX IF NOT EXISTS idx_search_analytics_query_hash 
ON search_analytics (query_hash);

CREATE INDEX IF NOT EXISTS idx_search_analytics_timestamp 
ON search_analytics (timestamp);

-- Table pour le cache des requêtes fréquentes
CREATE TABLE IF NOT EXISTS query_cache (
    query_hash VARCHAR(64) PRIMARY KEY,
    query_text TEXT NOT NULL,
    results JSONB,
    hit_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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

-- Vue pour les statistiques de performance
CREATE OR REPLACE VIEW search_performance_stats AS
SELECT 
    search_type,
    COUNT(*) as total_searches,
    AVG(response_time_ms) as avg_response_time,
    AVG(results_count) as avg_results_count,
    DATE_TRUNC('hour', timestamp) as hour_bucket
FROM search_analytics
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY search_type, DATE_TRUNC('hour', timestamp)
ORDER BY hour_bucket DESC;

-- Fonction de similarité personnalisée pour le ranking
CREATE OR REPLACE FUNCTION calculate_medical_similarity(
    embedding1 vector(768),
    embedding2 vector(768),
    boost_factor FLOAT DEFAULT 1.0
) RETURNS FLOAT AS $$
BEGIN
    -- Similarité cosinus avec boost optionnel
    RETURN (1 - (embedding1 <=> embedding2)) * boost_factor;
END;
$$ LANGUAGE plpgsql;

-- Fonction pour nettoyer les anciens caches
CREATE OR REPLACE FUNCTION cleanup_old_cache() RETURNS void AS $$
BEGIN
    -- Suppression des entrées de cache anciennes (> 1 semaine)
    DELETE FROM query_cache 
    WHERE last_used < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    -- Suppression des analytics anciennes (> 30 jours)
    DELETE FROM search_analytics 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Planification du nettoyage automatique (nécessite pg_cron)
-- SELECT cron.schedule('cleanup-medical-cache', '0 2 * * *', 'SELECT cleanup_old_cache();');

-- Insertion de données de test pour validation
INSERT INTO document_embeddings (document_id, sentence_id, content, embedding) VALUES
('test-doc-1', 'test-doc-1_0', 'Le patient présente une douleur thoracique depuis 2 jours.', 
 array_fill(0.1, ARRAY[768])::vector(768))
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
    END as status
UNION ALL
SELECT 
    'HNSW index' as component,
    CASE WHEN EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embeddings_vector')
         THEN 'CREATED'
         ELSE 'MISSING'
    END as status;