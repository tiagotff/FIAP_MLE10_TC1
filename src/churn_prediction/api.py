"""API de inferência do modelo de previsão de churn.

Endpoints:
- `POST /predict`: recebe os dados de um cliente e retorna a probabilidade
  de churn, a predição binária e o nível de risco.
- `GET /health`: verifica se a API está no ar e se o modelo foi carregado
  com sucesso (útil para liveness/readiness probes em produção).

Inclui middleware de latência (mede e loga o tempo de cada requisição) e
logging estruturado em todas as rotas (sem uso de `print()`).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from churn_prediction.inference import ModelNotLoadedError, predictor
from churn_prediction.logging_config import get_logger
from churn_prediction.schemas import ChurnPredictionRequest, ChurnPredictionResponse, HealthResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - assinatura exigida pelo FastAPI
    """Carrega o modelo uma única vez, na inicialização da aplicação."""
    logger.info("Iniciando API de inferência de churn")
    try:
        predictor.load()
    except FileNotFoundError as exc:
        # A API ainda sobe (permite que /health reporte o problema), mas
        # qualquer chamada a /predict falhará até o modelo ser treinado.
        logger.error("Falha ao carregar o modelo na inicialização", extra={"error": str(exc)})
    yield
    logger.info("Encerrando API de inferência de churn")


app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para o modelo de previsão de churn (Tech Challenge 1).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def latency_logging_middleware(request: Request, call_next):
    """Mede a latência de cada requisição e a registra em log estruturado.

    Também atribui um `request_id` único por requisição, propagado no log
    e no header de resposta — facilita rastrear uma requisição específica
    em um cenário real de observabilidade.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "Requisição processada",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
        },
    )

    return response


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):  # noqa: ARG001
    logger.error("Tentativa de predição com modelo não carregado", extra={"error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Modelo indisponível. Tente novamente em alguns instantes."},
    )


@app.get("/health", response_model=HealthResponse, tags=["observability"])
async def health() -> HealthResponse:
    """Verifica a saúde da API e se o modelo está carregado e pronto para uso."""
    if predictor.is_loaded:
        return HealthResponse(status="ok", model_loaded=True, model_version=predictor.model_version)
    return HealthResponse(status="degraded", model_loaded=False, model_version=None)


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["inference"])
async def predict(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Recebe os dados de um cliente e retorna a predição de risco de churn."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique se o treino foi executado ('make train').",
        )

    return predictor.predict(payload)
