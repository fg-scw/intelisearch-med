echo "🔬 Tests directs des APIs"
echo "========================"

# Test direct des APIs pour confirmer qu'elles fonctionnent
echo "1. Test health check search-api:"
curl -s http://localhost:8000/health | jq '.'

echo -e "\n2. Test health check processing:"
curl -s http://localhost:8001/health | jq '.'

echo -e "\n3. Test debug embeddings (avec correction SQL):"
curl -s http://localhost:8000/debug/embeddings | jq '.'

echo -e "\n4. Test upload direct:"
echo "Test content" > /tmp/test.txt
curl -X POST "http://localhost:8001/upload" \
  -F "file=@/tmp/test.txt" | jq '.'

echo -e "\n5. Test recherche après upload:"
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "search_type": "hybrid", "limit": 3}' | jq '.'