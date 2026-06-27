"""API de inferência do modelo de previsão de churn.

Endpoints:
- `GET /`: informações básicas da API (nome, versão, links úteis).
- `GET /health`: verifica se a API está no ar, se o modelo foi carregado
  com sucesso, e expõe um resumo das métricas do modelo (liveness/readiness
  probe + observabilidade de modelo em produção).
- `POST /predict`: recebe os dados de um cliente e retorna a probabilidade
  de churn, a predição binária e o nível de risco.
- `POST /predict/batch`: mesma predição, para até 500 clientes por chamada.

Inclui middleware de latência (mede e loga o tempo de cada requisição),
CORS habilitado (para consumo a partir de um frontend/browser) e logging
estruturado em todas as rotas (sem uso de `print()`). Qualquer exceção não
tratada é capturada por um handler genérico, que nunca expõe detalhes
internos (stack trace, mensagem de exceção) ao cliente da API.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from churn_prediction.inference import ModelNotLoadedError, predictor
from churn_prediction.logging_config import get_logger
from churn_prediction.schemas import (
    BatchChurnPredictionRequest,
    BatchChurnPredictionResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
    RootResponse,
)

logger = get_logger(__name__)

API_VERSION = "1.0.0"


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
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS: permite que um frontend (dashboard de CRM, por exemplo) consuma a
# API diretamente do navegador. Em um cenário real de produção, a lista de
# origens permitidas deveria ser restrita aos domínios confiáveis em vez de
# "*" — mantido amplo aqui por se tratar de um projeto de portfólio/estudo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Captura qualquer exceção não tratada, evitando expor detalhes internos
    (stack trace, tipo da exceção, mensagem original) ao cliente da API —
    boa prática de segurança e de UX de API. O erro completo continua sendo
    registrado no log estruturado para investigação.
    """
    logger.exception(
        "Erro não tratado ao processar requisição",
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno ao processar a requisição. Tente novamente."},
    )


@app.get("/", response_model=RootResponse, tags=["observability"])
async def root() -> RootResponse:
    """Informações básicas da API — evita um 404 "vazio" na rota raiz."""
    return RootResponse(
        name="Churn Prediction API",
        version=API_VERSION,
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.get("/health", response_model=HealthResponse, tags=["observability"])
async def health() -> HealthResponse:
    """Verifica a saúde da API e se o modelo está carregado e pronto para uso.

    Quando o modelo está carregado, inclui também um resumo de suas
    métricas (AUC-ROC, recall, custo de negócio no holdout de teste),
    útil para verificação rápida de qual versão/qualidade de modelo está
    em produção sem precisar consultar o MLflow.
    """
    if predictor.is_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=predictor.model_version,
            model_info=predictor.get_model_info(),
        )
    return HealthResponse(status="degraded", model_loaded=False, model_version=None, model_info=None)


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["inference"])
async def predict(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Recebe os dados de um cliente e retorna a predição de risco de churn."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique se o treino foi executado ('make train').",
        )

    return predictor.predict(payload)


@app.post("/predict/batch", response_model=BatchChurnPredictionResponse, tags=["inference"])
async def predict_batch(payload: BatchChurnPredictionRequest) -> BatchChurnPredictionResponse:
    """Recebe uma lista de clientes (até 500) e retorna a predição de cada um.

    Útil para rotinas batch do time de Retenção/CRM pontuarem uma carteira
    de clientes de uma vez, evitando uma chamada HTTP por cliente.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique se o treino foi executado ('make train').",
        )

    predictions = predictor.predict_batch(payload.customers)
    return BatchChurnPredictionResponse(predictions=predictions, count=len(predictions))
