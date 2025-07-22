# ============================================================================
# SCRIPT DE DIAGNOSTIC COMPLET
# ============================================================================

#!/bin/bash
echo "🔍 Diagnostic des problèmes de connectivité"
echo "==========================================="

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${BLUE}1. Vérification de l'état des services${NC}"
echo "---------------------------------------"
docker-compose ps

echo -e "\n${BLUE}2. Test des endpoints de santé${NC}"
echo "--------------------------------"

echo "Processing service (8001):"
curl -v http://localhost:8001/health 2>&1 | head -10

echo -e "\nSearch API (8000):"
curl -v http://localhost:8000/health 2>&1 | head -10

echo -e "\nFrontend (3000):"
curl -v http://localhost:3000 2>&1 | head -10

echo -e "\n${BLUE}3. Vérification des logs des services${NC}"
echo "------------------------------------"
echo "Logs Processing (dernières 10 lignes):"
docker-compose logs --tail=10 processing-service

echo -e "\nLogs Search API (dernières 10 lignes):"
docker-compose logs --tail=10 search-api

echo -e "\n${BLUE}4. Test de connectivité réseau Docker${NC}"
echo "-----------------------------------"
echo "Test depuis le container frontend vers search-api:"
docker-compose exec frontend wget -qO- http://search-api:8000/health || echo "❌ Connexion échouée"

echo -e "\nTest depuis le container frontend vers processing:"
docker-compose exec frontend wget -qO- http://processing-service:8001/health || echo "❌ Connexion échouée"

echo -e "\n${BLUE}5. Vérification des embeddings en base${NC}"
echo "---------------------------------------------"
echo "Nombre d'embeddings stockés:"
docker-compose exec postgres psql -U medical_user -d medical_search -c "SELECT COUNT(*) FROM document_embeddings;"

echo -e "\nÉchantillon d'embeddings:"
docker-compose exec postgres psql -U medical_user -d medical_search -c "SELECT document_id, content, array_length(embedding, 1) as emb_size FROM document_embeddings LIMIT 3;"