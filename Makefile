.PHONY: help install lint format test test-cov train run clean

PYTHON := python3
PYTHONPATH := src

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Instala o projeto e dependências de desenvolvimento
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Roda o linter (ruff) em todo o projeto
	ruff check .

format: ## Formata o código automaticamente (ruff --fix)
	ruff check --fix .

test: ## Roda a suíte de testes automatizados (pytest)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

test-cov: ## Roda os testes com relatório de cobertura
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v --cov=src/churn_prediction --cov-report=term-missing

train: ## Treina o modelo de produção (MLP) e salva os artefatos em models/
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m churn_prediction.train

run: ## Inicia a API de inferência localmente (http://127.0.0.1:8000)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn churn_prediction.api:app --host 127.0.0.1 --port 8000 --reload

clean: ## Remove caches e artefatos gerados (não remove models/ treinados)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
