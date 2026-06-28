"""Model registry: garante que os artefatos do modelo existam localmente
antes da inferência, baixando-os de um bucket do Google Cloud Storage
quando configurado.

Design: este módulo é **opcional e não-invasivo**. Se a variável de
ambiente `MODEL_BUCKET` não estiver definida, `ensure_model_artifacts()`
não faz nada — o comportamento permanece idêntico ao de desenvolvimento
local (Etapas 1-3): os artefatos já devem existir em `models/`, gerados
por `python -m churn_prediction.train`.

Quando `MODEL_BUCKET` está definido (cenário de deploy em nuvem, Etapa 4),
os artefatos são baixados do bucket para `models/` na inicialização da
API, antes do `ChurnPredictor.load()` ser chamado — desacoplando o ciclo
de vida do treino (que pode rodar a qualquer momento, em qualquer
máquina) do ciclo de vida do deploy da API (que não precisa ser refeito
só porque um novo modelo foi treinado).
"""

from __future__ import annotations

import os

from churn_prediction.config import MODELS_DIR
from churn_prediction.logging_config import get_logger

logger = get_logger(__name__)

# Nome dos artefatos esperados no bucket, dentro do prefixo configurado.
ARTIFACT_FILENAMES = ("mlp_model.pt", "preprocessor.joblib", "model_metadata.json")

MODEL_BUCKET_ENV_VAR = "MODEL_BUCKET"
MODEL_BUCKET_PREFIX_ENV_VAR = "MODEL_BUCKET_PREFIX"
DEFAULT_BUCKET_PREFIX = "models"


def ensure_model_artifacts() -> None:
    """Garante que os artefatos do modelo existam em `MODELS_DIR`.

    Se `MODEL_BUCKET` não estiver configurado, não faz nada (assume que os
    artefatos já existem localmente, fluxo padrão de desenvolvimento).
    Se estiver configurado, baixa cada artefato do bucket para `MODELS_DIR`,
    sobrescrevendo qualquer versão local existente — o bucket é tratado
    como a fonte de verdade nesse modo.
    """
    bucket_name = os.environ.get(MODEL_BUCKET_ENV_VAR)
    if not bucket_name:
        logger.info(
            "MODEL_BUCKET não configurado — usando artefatos locais em models/",
        )
        return

    prefix = os.environ.get(MODEL_BUCKET_PREFIX_ENV_VAR, DEFAULT_BUCKET_PREFIX)

    # Import local (lazy): evita exigir 'google-cloud-storage' instalado em
    # ambientes que nunca usam o modo de bucket (ex.: execução local padrão).
    from google.cloud import storage

    logger.info(
        "Baixando artefatos do modelo a partir do Cloud Storage",
        extra={"bucket": bucket_name, "prefix": prefix},
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for filename in ARTIFACT_FILENAMES:
        blob_path = f"{prefix}/{filename}"
        blob = bucket.blob(blob_path)
        destination = MODELS_DIR / filename

        if not blob.exists():
            # model_metadata.json é opcional (a API funciona sem ele, só
            # perde o resumo de métricas em /metadata) — os dois outros
            # arquivos são obrigatórios e o erro vai aparecer naturalmente
            # no ChurnPredictor.load() caso estejam ausentes.
            logger.warning(
                "Artefato não encontrado no bucket, pulando",
                extra={"bucket": bucket_name, "blob_path": blob_path},
            )
            continue

        blob.download_to_filename(str(destination))
        logger.info(
            "Artefato baixado com sucesso",
            extra={"blob_path": blob_path, "destination": str(destination)},
        )
