#!/bin/bash
# setup_project.sh - Script de setup complet du projet

echo "🏥 Setup du Pipeline de Recherche Sémantique Médicale"
echo "===================================================="

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n${BLUE}1. Création de la structure de répertoires${NC}"
echo "----------------------------------------"

# Création des répertoires principaux
mkdir -p services/{processing,search-api,frontend,init}
mkdir -p services/frontend/{src,public}
mkdir -p config init-scripts data/{uploads,processed} models demo/test_documents

echo "✅ Structure de répertoires créée"

echo -e "\n${BLUE}2. Création des fichiers de configuration${NC}"
echo "--------------------------------------------"

# Frontend - nginx.conf
cat > services/frontend/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    server {
        listen 3000;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html index.htm;

        location / {
            try_files $uri $uri/ /index.html;
        }

        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
EOF

# Frontend - package.json
cat > services/frontend/package.json << 'EOF'
{
  "name": "medical-search-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version"]
  },
  "homepage": "."
}
EOF

# Frontend - public/index.html
cat > services/frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <title>Recherche Sémantique Médicale</title>
</head>
<body>
    <noscript>JavaScript requis</noscript>
    <div id="root"></div>
</body>
</html>
EOF

# Frontend - src/index.js
cat > services/frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

echo "✅ Fichiers de configuration créés"

echo -e "\n${BLUE}3. Configuration OpenSearch${NC}"
echo "------------------------------"

cat > config/opensearch.yml << 'EOF'
cluster.name: medical-search-cluster
node.name: medical-search-node
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
cluster.initial_master_nodes: ["medical-search-node"]
bootstrap.memory_lock: true
plugins.security.disabled: true
EOF

echo "✅ Configuration OpenSearch créée"

echo -e "\n${BLUE}4. Création des requirements Python${NC}"
echo "--------------------------------------"

# Requirements processing
cat > services/processing/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
asyncpg==0.29.0
aioredis==2.0.1
aiofiles==23.2.1
opensearch-py==2.4.2
sentence-transformers==2.2.2
transformers==4.35.2
torch==2.1.1
numpy==1.24.3
PyPDF2==3.0.1
python-multipart==0.0.6
pydantic==2.5.0
EOF

# Requirements search-api
cat > services/search-api/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
asyncpg==0.29.0
aioredis==2.0.1
opensearch-py==2.4.2
sentence-transformers==2.2.2
transformers==4.35.2
torch==2.1.1
numpy==1.24.3
pydantic==2.5.0
EOF

echo "✅ Requirements Python créés"

echo -e "\n${BLUE}5. Script d'initialisation simple${NC}"
echo "------------------------------------"

cat > services/init/init.py << 'EOF'
import asyncio
import asyncpg
import os
import logging
from opensearchpy import AsyncOpenSearch
import aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def wait_for_services():
    """Attendre que les services soient prêts"""
    await asyncio.sleep(15)  # Attendre 15 secondes

async def init_opensearch():
    """Initialisation basique d'OpenSearch"""
    try:
        client = AsyncOpenSearch(
            hosts=[os.getenv('OPENSEARCH_URL', 'http://opensearch:9200')],
            use_ssl=False,
            verify_certs=False
        )
        
        # Créer l'index s'il n'existe pas
        if not await client.indices.exists(index="medical_documents"):
            await client.indices.create(
                index="medical_documents",
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "title": {"type": "text"},
                            "file_type": {"type": "keyword"}
                        }
                    }
                }
            )
            logger.info("Index OpenSearch créé")
        await client.close()
    except Exception as e:
        logger.error(f"Erreur OpenSearch: {e}")

async def init_postgres():
    """Test de connexion PostgreSQL"""
    try:
        conn = await asyncpg.connect(os.getenv('POSTGRES_URL'))
        result = await conn.fetchval("SELECT 1")
        logger.info(f"PostgreSQL connecté: {result}")
        await conn.close()
    except Exception as e:
        logger.error(f"Erreur PostgreSQL: {e}")

async def main():
    logger.info("Initialisation des services...")
    await wait_for_services()
    await asyncio.gather(
        init_opensearch(),
        init_postgres(),
        return_exceptions=True
    )
    logger.info("Initialisation terminée")

if __name__ == "__main__":
    asyncio.run(main())
EOF

echo "✅ Script d'initialisation créé"

echo -e "\n${BLUE}6. Création des .gitkeep pour les dossiers vides${NC}"
echo "------------------------------------------------"

touch data/uploads/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep

echo "✅ Fichiers .gitkeep créés"

echo -e "\n${BLUE}7. Documents de test${NC}"
echo "--------------------"

cat > demo/test_documents/cardiologie.txt << 'EOF'
PATIENT: Jean Dupont, 65 ans

MOTIF DE CONSULTATION:
Douleur thoracique depuis 2 heures

HISTOIRE DE LA MALADIE:
Douleur constrictive rétrosternale irradiant vers le bras gauche.
Apparition brutale au repos.
Sueurs, nausées associées.

ANTÉCÉDENTS:
- HTA connue sous traitement
- Tabagisme sevré depuis 5 ans
- Dyslipidémie

EXAMEN CLINIQUE:
TA: 160/95 mmHg, FC: 95/min
Auscultation cardiaque: normal
Auscultation pulmonaire: normal

EXAMENS COMPLÉMENTAIRES:
ECG: sus-décalage ST en antérieur (D1, aVL, V1-V6)
Troponines I: 15 ng/ml (N < 0.1)

DIAGNOSTIC:
Syndrome coronarien aigu ST+
Infarctus du myocarde antérieur

TRAITEMENT:
Aspégic 250mg IV
Plavix 600mg charge puis 75mg/j
Angioplastie primaire urgente
EOF

cat > demo/test_documents/pneumologie.txt << 'EOF'
PATIENTE: Marie Martin, 52 ans

MOTIF DE CONSULTATION:
Dyspnée d'effort progressive depuis 6 mois

HISTOIRE DE LA MALADIE:
Essoufflement à l'effort devenu invalidant
Toux sèche chronique
Fatigue importante

ANTÉCÉDENTS:
- Tabagisme actif: 35 paquets-années
- Pas d'exposition professionnelle connue

EXAMEN CLINIQUE:
FR: 22/min, SaO2: 92% air ambiant
Auscultation: diminution du murmure vésiculaire
Râles crépitants bilatéraux

IMAGERIE:
Scanner thoracique:
- Emphysème pan-lobulaire diffus
- Distension thoracique
- Cœur pulmonaire chronique débutant

EFR:
VEMS: 42% de la théorique
CVF: 78% de la théorique
VEMS/CVF: 0.45

DIAGNOSTIC:
BPCO stade sévère (GOLD 3)
Emphysème pan-lobulaire

TRAITEMENT:
Seretide 125/25: 2 bouffées matin et soir
Spiriva 18mcg: 1 gélule/jour
Sevrage tabagique urgent
Réhabilitation respiratoire
EOF

echo "✅ Documents de test créés"

echo -e "\n${GREEN}✅ Setup terminé avec succès!${NC}"
echo -e "\n${YELLOW}Prochaines étapes:${NC}"
echo "1. Vérifiez que Docker Desktop est lancé"
echo "2. Nettoyez les conteneurs précédents:"
echo -e "   ${BLUE}docker-compose down -v${NC}"
echo "3. Construisez les images:"
echo -e "   ${BLUE}docker-compose build${NC}"
echo "4. Lancez les services:"
echo -e "   ${BLUE}docker-compose up -d${NC}"
echo "5. Surveillez les logs:"
echo -e "   ${BLUE}docker-compose logs -f${NC}"
echo -e "\n${YELLOW}Interfaces disponibles après démarrage:${NC}"
echo -e "- Interface web: ${BLUE}http://localhost:3000${NC}"
echo -e "- API recherche: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "- Service processing: ${BLUE}http://localhost:8001/docs${NC}"
echo -e "- OpenSearch: ${BLUE}http://localhost:9200${NC}"