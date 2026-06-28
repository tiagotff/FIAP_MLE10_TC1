# syntax=docker/dockerfile:1

# Dockerfile de produção para a API de inferência de churn.
#
# Os artefatos do modelo (mlp_model.pt, preprocessor.joblib) NÃO são
# copiados para a imagem — são baixados em runtime de um bucket do Google
# Cloud Storage (ver src/churn_prediction/model_registry.py), configurado
# via a variável de ambiente MODEL_BUCKET no serviço Cloud Run. Isso
# desacopla o ciclo de vida do treino (pode ser refeito a qualquer
# momento) do ciclo de vida do deploy de código — um novo modelo treinado
# não exige rebuild/redeploy da imagem, só atualizar o conteúdo do bucket.
#
# Outras otimizações:
# - Instala o PyTorch CPU-only (índice oficial), evitando ~1.5GB de
#   dependências CUDA desnecessárias (o Cloud Run não tem GPU).
# - Usa apenas requirements-api.txt (não o ambiente de dev completo do
#   pyproject.toml), reduzindo o tamanho final da imagem.

FROM python:3.12-slim AS base

WORKDIR /app

# Dependências de sistema mínimas (libgomp é exigida pelo PyTorch/sklearn).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala o PyTorch CPU-only primeiro (camada cacheável, raramente muda).
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

# Copia apenas o código-fonte da aplicação — nenhum artefato de modelo,
# dataset, notebook, teste ou credencial entra na imagem.
COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# O Cloud Run injeta a variável PORT em runtime; 8080 é o padrão esperado.
ENV PORT=8080
EXPOSE 8080

# Usuário não-root, por boa prática de segurança de containers.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# shell form: permite a expansão da variável $PORT injetada pelo Cloud Run.
CMD uvicorn churn_prediction.api:app --host 0.0.0.0 --port ${PORT}
