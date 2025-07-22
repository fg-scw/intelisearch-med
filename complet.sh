#!/bin/bash
echo "🧪 Test complet après corrections"
echo "================================"

# Upload de documents de test
echo "1. Upload des documents de test..."
curl -X POST "http://localhost:8001/upload" \
  -F "file=@demo/test_documents/cardiologie.txt" | jq '.document_id'

curl -X POST "http://localhost:8001/upload" \
  -F "file=@demo/test_documents/pneumologie.txt" | jq '.document_id'

# Attendre un peu pour le traitement
sleep 5

echo -e "\n2. Debug des embeddings après upload:"
curl -s "http://localhost:8000/debug/embeddings" | jq '{total_embeddings, unique_documents, null_embeddings}'

echo -e "\n3. Test recherche textuelle:"
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "douleur thoracique", "search_type": "textual", "limit": 3}' | jq '.[].title'

echo -e "\n4. Test recherche sémantique:"
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "cardiologie", "search_type": "semantic", "limit": 3}' | jq '.[].score'

echo -e "\n5. Test recherche hybride:"
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "cardiologie", "search_type": "hybrid", "limit": 3}' | jq '.[] | {title, hybrid_score, relevance_score, semantic_score}'

echo -e "\n6. Statistiques finales:"
curl -s "http://localhost:8000/stats" | jq '.'

echo -e "\n✅ Tests terminés!"
echo "🌐 Interface web: http://localhost:3000"
