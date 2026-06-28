#!/usr/bin/env bash
# Builda e implanta o dashboard Streamlit no Cloud Run, configurado para
# consumir a API de inferência já implantada.
#
# Uso:
#   ./scripts/deploy_streamlit_to_cloud_run.sh <projeto-gcp> <url-da-api> [regiao]
#
# Exemplo:
#   ./scripts/deploy_streamlit_to_cloud_run.sh churn-prediction-tc1 \
#       https://churn-api-855490327597.us-central1.run.app
#
# Pré-requisitos:
#   - gcloud CLI instalado e autenticado (gcloud auth login)
#   - APIs run.googleapis.com e cloudbuild.googleapis.com habilitadas
#   - A API de inferência já implantada (ver README.md / deployment_architecture.md)
#   - Rodar a partir da raiz do projeto (onde está o cloudbuild.streamlit.yaml)

set -euo pipefail

PROJECT_ID="${1:-}"
API_URL="${2:-}"
REGION="${3:-us-central1}"

if [[ -z "${PROJECT_ID}" || -z "${API_URL}" ]]; then
  echo "Uso: $0 <projeto-gcp> <url-da-api> [regiao]" >&2
  exit 1
fi

IMAGE="gcr.io/${PROJECT_ID}/churn-streamlit"

echo "Buildando a imagem do dashboard Streamlit (${IMAGE})..."
gcloud builds submit \
  --config cloudbuild.streamlit.yaml \
  --substitutions="_IMAGE=${IMAGE}"

echo "Implantando no Cloud Run..."
gcloud run deploy churn-streamlit \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --set-env-vars "CHURN_API_URL=${API_URL}" \
  --memory 512Mi \
  --cpu 1 \
  --port 8080

echo "Deploy concluído."
