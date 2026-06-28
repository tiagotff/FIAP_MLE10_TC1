#!/usr/bin/env bash
# Sobe os artefatos do modelo treinado (models/) para o bucket do Cloud
# Storage configurado como model registry da API em produção.
#
# Uso:
#   ./scripts/upload_model_to_gcs.sh <nome-do-bucket> [prefixo]
#
# Exemplo:
#   ./scripts/upload_model_to_gcs.sh meu-bucket-churn-model
#   ./scripts/upload_model_to_gcs.sh meu-bucket-churn-model models/v2
#
# Pré-requisitos:
#   - gcloud CLI instalado e autenticado (gcloud auth login)
#   - Artefatos já gerados localmente via 'make train' (ou
#     'python -m churn_prediction.train')
#   - O bucket já deve existir (ver README.md / deployment_architecture.md
#     para o comando de criação)

set -euo pipefail

BUCKET_NAME="${1:-}"
PREFIX="${2:-models}"

if [[ -z "${BUCKET_NAME}" ]]; then
  echo "Uso: $0 <nome-do-bucket> [prefixo]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${PROJECT_ROOT}/models"

REQUIRED_FILES=("mlp_model.pt" "preprocessor.joblib")
OPTIONAL_FILES=("model_metadata.json")

for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${MODELS_DIR}/${file}" ]]; then
    echo "ERRO: ${MODELS_DIR}/${file} não encontrado." >&2
    echo "Execute 'make train' (ou o comando equivalente) antes de subir o modelo." >&2
    exit 1
  fi
done

echo "Subindo artefatos para gs://${BUCKET_NAME}/${PREFIX}/ ..."

for file in "${REQUIRED_FILES[@]}" "${OPTIONAL_FILES[@]}"; do
  filepath="${MODELS_DIR}/${file}"
  if [[ -f "${filepath}" ]]; then
    gsutil cp "${filepath}" "gs://${BUCKET_NAME}/${PREFIX}/${file}"
  else
    echo "Aviso: ${filepath} não encontrado, pulando (arquivo opcional)." >&2
  fi
done

echo "Upload concluído."
echo "Para usar este bucket na API, configure no Cloud Run:"
echo "  MODEL_BUCKET=${BUCKET_NAME}"
echo "  MODEL_BUCKET_PREFIX=${PREFIX}"
