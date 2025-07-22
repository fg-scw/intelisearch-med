# Makefile
# Commandes utiles pour le projet

.PHONY: help build up down logs clean test

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Construit les images Docker
	docker-compose build

up: ## Lance tous les services
	docker-compose up -d

down: ## Arrête tous les services
	docker-compose down

logs: ## Affiche les logs de tous les services
	docker-compose logs -f

clean: ## Nettoie les volumes et images
	docker-compose down -v
	docker system prune -f

test: ## Exécute les tests du pipeline
	bash demo/test_pipeline.sh

setup: ## Première installation complète
	mkdir -p data/uploads data/processed models
	mkdir -p demo/test_documents
	@echo "Création de la structure de fichiers terminée"
	@echo "Lancez 'make build' puis 'make up' pour démarrer"