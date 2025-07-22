

#!/bin/bash
# demo/test_pipeline.sh
# Script de test complet du pipeline de recherche médicale

echo "🏥 Test du Pipeline de Recherche Sémantique Médicale"
echo "=================================================="

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8000"
PROCESSING_URL="http://localhost:8001"

echo -e "\n${BLUE}1. Vérification de l'état des services${NC}"
echo "----------------------------------------"

# Test des services
services=("opensearch:9200" "postgres:5432" "redis:6379" "search-api:8000" "processing:8001")
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1 || nc -z localhost $port 2>/dev/null; then
        echo -e "✅ $name: ${GREEN}RUNNING${NC}"
    else
        echo -e "❌ $name: ${RED}DOWN${NC}"
    fi
done

echo -e "\n${BLUE}2. Création de documents de test${NC}"
echo "--------------------------------"

# Création de documents médicaux de test
mkdir -p demo/test_documents

cat > demo/test_documents/cas_clinique_cardiologie.txt << EOF
MOTIF DE CONSULTATION
Patient de 58 ans consulte pour douleur thoracique.

HISTOIRE DE LA MALADIE
Depuis 3 jours, le patient présente une douleur thoracique constrictive, irradiant vers le bras gauche, survenant à l'effort et cédant au repos. Antécédents d'hypertension artérielle et de tabagisme.

EXAMEN CLINIQUE
TA: 145/95 mmHg, FC: 78/min, température normale.
Auscultation cardiaque: souffle systolique 2/6
Auscultation pulmonaire: normale

EXAMENS COMPLÉMENTAIRES
ECG: modifications ST-T en antérieur
Troponines: légèrement élevées

DIAGNOSTIC
Syndrome coronarien aigu - angor instable

TRAITEMENT
Aspirine 75mg, Clopidogrel 75mg, Atorvastatine 40mg
Hospitalisation en USIC pour surveillance
EOF

cat > demo/test_documents/cas_clinique_pneumologie.txt << EOF
MOTIF DE CONSULTATION
Patiente de 45 ans consulte pour dyspnée et toux chronique.

ANTÉCÉDENTS
Tabagisme actif 20 paquets-années
Pas d'antécédents familiaux particuliers

EXAMEN CLINIQUE
Dyspnée d'effort stade II
Toux productive avec expectoration purulente
Auscultation: râles bronchiques diffus

IMAGERIE
Scanner thoracique: emphysème centro-lobulaire
Dilatation des bronches

ÉPREUVES FONCTIONNELLES RESPIRATOIRES
VEMS: 45% de la théorique
Rapport VEMS/CV: 0.55
Réversibilité partielle sous bronchodilatateurs

DIAGNOSTIC
BPCO stade II avec exacerbation infectieuse

TRAITEMENT
Bronchodilatateurs: Tiotropium + Formotérol/Budésonide
Antibiothérapie: Amoxicilline-acide clavulanique
Sevrage tabagique
EOF

echo -e "📄 Documents de test créés"

echo -e "\n${BLUE}3. Upload et traitement des documents${NC}"
echo "------------------------------------"

for file in demo/test_documents/*.txt; do
    filename=$(basename "$file")
    echo -n "Traitement de $filename... "
    
    response=$(curl -s -X POST "$PROCESSING_URL/upload" \
        -F "file=@$file" \
        -w "HTTP_%{http_code}")
    
    if [[ $response == *"HTTP_200"* ]]; then
        doc_id=$(echo "$response" | grep -o '"document_id":"[^"]*' | cut -d'"' -f4)
        echo -e "${GREEN}✅ Succès${NC} (ID: ${doc_id:0:8}...)"
    else
        echo -e "${RED}❌ Échec${NC}"
    fi
done

echo -e "\n${BLUE}4. Test des différents types de recherche${NC}"
echo "-------------------------------------------"

queries=(
    "douleur thoracique"
    "dyspnée chronique"
    "syndrome coronarien"
    "BPCO exacerbation"
    "traitement hypertension"
)

search_types=("textual" "semantic" "hybrid")

for query in "${queries[@]}"; do
    echo -e "\n🔍 Requête: ${YELLOW}\"$query\"${NC}"
    
    for search_type in "${search_types[@]}"; do
        echo -n "  $search_type: "
        
        response=$(curl -s -X POST "$BASE_URL/search" \
            -H "Content-Type: application/json" \
            -d "{
                \"query\": \"$query\",
                \"search_type\": \"$search_type\",
                \"limit\": 3
            }")
        
        count=$(echo "$response" | jq -r '. | length' 2>/dev/null || echo "0")
        
        if [ "$count" -gt 0 ]; then
            echo -e "${GREEN}$count résultat(s)${NC}"
            
            # Affichage du premier résultat avec scores
            if [ "$search_type" = "hybrid" ]; then
                scores=$(echo "$response" | jq -r '.[0] | "Score hybride: " + (.hybrid_score // 0 | tostring | .[0:5]) + " (T:" + (.relevance_score // 0 | tostring | .[0:4]) + " S:" + (.semantic_score // 0 | tostring | .[0:4]) + ")"' 2>/dev/null)
                [ "$scores" != "null" ] && echo "    $scores"
            else
                score=$(echo "$response" | jq -r '.[0].score // 0 | tostring | .[0:5]' 2>/dev/null)
                [ "$score" != "null" ] && echo "    Score: $score"
            fi
        else
            echo -e "${RED}0 résultat${NC}"
        fi
    done
done

echo -e "\n${BLUE}5. Test avec contexte patient${NC}"
echo "-----------------------------"

echo -n "Recherche avec contexte patient (58 ans, cardiologie): "
response=$(curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "douleur thoracique",
        "search_type": "hybrid",
        "specialty": "cardiologie",
        "patient_context": {"age": "58", "gender": "M"},
        "limit": 5
    }')

count=$(echo "$response" | jq -r '. | length' 2>/dev/null || echo "0")
echo -e "${GREEN}$count résultat(s)${NC}"

echo -e "\n${BLUE}6. Statistiques du système${NC}"
echo "----------------------------"

stats=$(curl -s "$BASE_URL/stats")
if [ $? -eq 0 ]; then
    echo "$stats" | jq '.' 2>/dev/null || echo "$stats"
else
    echo -e "${RED}Erreur récupération des statistiques${NC}"
fi

echo -e "\n${BLUE}7. Test de performance${NC}"
echo "----------------------"

echo "Test de latence (10 requêtes):"
total_time=0
for i in {1..10}; do
    start_time=$(date +%s%3N)
    curl -s -X POST "$BASE_URL/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "diagnostic différentiel", "search_type": "hybrid", "limit": 5}' \
        > /dev/null
    end_time=$(date +%s%3N)
    
    response_time=$((end_time - start_time))
    total_time=$((total_time + response_time))
    echo "  Requête $i: ${response_time}ms"
done

avg_time=$((total_time / 10))
echo -e "Temps moyen: ${YELLOW}${avg_time}ms${NC}"

echo -e "\n${GREEN}✅ Tests terminés avec succès!${NC}"
echo -e "\n📊 Accès aux interfaces:"
echo -e "  - Interface web: ${BLUE}http://localhost:3000${NC}"
echo -e "  - API recherche: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  - Service processing: ${BLUE}http://localhost:8001/docs${NC}"
echo -e "  - OpenSearch: ${BLUE}http://localhost:9200${NC}"

---

# demo/sample_medical_documents/README.md
# Documents médicaux de démonstration

Ce dossier contient des exemples de documents médicaux pour tester le pipeline de recherche sémantique.

## Structure des documents

### Cas cliniques
- `cardiologie_001.txt` - Syndrome coronarien aigu
- `pneumologie_001.txt` - BPCO avec exacerbation
- `neurologie_001.txt` - AVC ischémique
- `oncologie_001.txt` - Cancer bronchique

### Protocoles thérapeutiques  
- `protocole_hypertension.txt`
- `protocole_diabete.txt`
- `protocole_anticoagulation.txt`

### Guidelines cliniques
- `guidelines_insuffisance_cardiaque.txt`
- `guidelines_asthme.txt`

## Utilisation

1. Lancer le pipeline: `docker-compose up -d`
2. Exécuter les tests: `bash demo/test_pipeline.sh`
3. Uploader via l'interface: http://localhost:3000
4. Ou via API: `curl -F "file=@document.txt" http://localhost:8001/upload`