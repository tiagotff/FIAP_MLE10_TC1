"""API de inferência do modelo de previsão de churn.

Endpoints, seguindo a convenção de plataformas de model serving em produção
(KServe, Seldon, BentoML, MLflow Serving):

- `GET /health`: **liveness** — o processo da API está vivo? Não depende do
  modelo estar carregado; usado por um orquestrador (ex.: Kubernetes) para
  decidir se deve reiniciar o container.
- `GET /ready`: **readiness** — a API está pronta para receber tráfego?
  Verifica se o modelo e o pipeline foram carregados; usado por um load
  balancer para decidir se deve rotear requisições para esta instância.
- `POST /infer`: realiza a predição de churn para um cliente (nome canônico
  de inferência). `POST /predict` e `POST /predict/batch` continuam
  disponíveis como aliases, por compatibilidade com integrações existentes.
- `GET /metadata`: descreve o modelo em produção (versão, métricas) e o
  contrato de entrada/saída esperado por `/infer`.
- `GET /metrics`: métricas operacionais da API (contagem de requisições,
  latência, predições por nível de risco) no formato texto do Prometheus.
- `GET /`: informações básicas e links para os demais endpoints.

Inclui middleware de latência + métricas Prometheus (mede e registra o
tempo de cada requisição), CORS habilitado (para consumo a partir de um
frontend/browser) e logging estruturado em todas as rotas (sem uso de
`print()`). Qualquer exceção não tratada é capturada por um handler
genérico, que nunca expõe detalhes internos (stack trace, mensagem de
exceção) ao cliente da API.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from churn_prediction.inference import ModelNotLoadedError, predictor
from churn_prediction.logging_config import get_logger
from churn_prediction.metrics import observe_prediction, observe_request, render_latest_metrics
from churn_prediction.model_registry import ensure_model_artifacts
from churn_prediction.schemas import (
    BatchChurnPredictionRequest,
    BatchChurnPredictionResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
    MetadataResponse,
    ReadyResponse,
    RootResponse,
)

logger = get_logger(__name__)

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - assinatura exigida pelo FastAPI
    """Carrega o modelo uma única vez, na inicialização da aplicação.

    Se a variável de ambiente MODEL_BUCKET estiver configurada (cenário de
    deploy em nuvem), baixa os artefatos do Cloud Storage antes de
    carregá-los — ver `model_registry.py`. Em desenvolvimento local, esse
    passo é um no-op e os artefatos já existentes em models/ são usados
    diretamente.
    """
    logger.info("Iniciando API de inferência de churn")
    try:
        ensure_model_artifacts()
        predictor.load()
    except FileNotFoundError as exc:
        # A API ainda sobe (permite que /health responda e /ready reporte o
        # problema), mas qualquer chamada a /infer falhará até o modelo ser
        # treinado.
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
    """Mede a latência de cada requisição, registrando-a em log estruturado
    e nas métricas Prometheus (`/metrics`).

    Também atribui um `request_id` único por requisição, propagado no log
    e no header de resposta — facilita rastrear uma requisição específica
    em um cenário real de observabilidade.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    response = await call_next(request)

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_ms = elapsed_seconds * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    # request.url.path, e não a rota com path params resolvida, é
    # suficiente aqui pois esta API não tem rotas parametrizadas.
    observe_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_seconds=elapsed_seconds,
    )

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
        ready_url="/ready",
        metadata_url="/metadata",
        metrics_url="/metrics",
        infer_url="/infer",
    )


@app.get("/health", response_model=HealthResponse, tags=["observability"])
async def health() -> HealthResponse:
    """Liveness probe: confirma que o processo da API está respondendo.

    Não verifica o modelo — propositalmente simples e rápido, para que um
    orquestrador não reinicie o container por um problema no modelo (que é
    responsabilidade do `/ready`, não do `/health`).
    """
    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["observability"])
async def ready() -> ReadyResponse:
    """Readiness probe: confirma que o modelo e o pipeline estão carregados
    e a API está pronta para receber tráfego de inferência.

    Diferente de `/health`: a API pode estar viva (`/health` = ok) mas
    ainda não pronta (`/ready` = not_ready), por exemplo durante o
    carregamento inicial do modelo ou se o treino nunca foi executado.
    """
    if predictor.is_loaded:
        return ReadyResponse(status="ready", model_loaded=True)
    return ReadyResponse(status="not_ready", model_loaded=False)


@app.get("/metadata", response_model=MetadataResponse, tags=["observability"])
async def metadata() -> MetadataResponse:
    """Descreve o modelo em produção e o contrato de entrada/saída de `/infer`.

    Útil para um cliente (humano ou sistema) descobrir programaticamente
    como montar uma requisição válida e quais métricas o modelo atual tem,
    sem precisar consultar o MLflow ou ler a documentação manualmente.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique se o treino foi executado ('make train').",
        )

    model_info = predictor.get_model_info()
    if model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadados do modelo indisponíveis.",
        )

    return MetadataResponse(
        model_info=model_info,
        input_schema=ChurnPredictionRequest.model_json_schema(),
        output_schema=ChurnPredictionResponse.model_json_schema(),
    )


@app.get("/metrics", tags=["observability"])
async def metrics() -> Response:
    """Métricas operacionais da API (requisições, latência, predições por
    risco) no formato texto do Prometheus, para scraping por um servidor
    Prometheus ou visualização em um dashboard Grafana.
    """
    body, content_type = render_latest_metrics()
    return Response(content=body, media_type=content_type)


def _predict_and_track(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Executa a predição e registra o nível de risco resultante nas métricas."""
    result = predictor.predict(payload)
    observe_prediction(result.risk_level)
    return result


@app.post("/infer", response_model=ChurnPredictionResponse, tags=["inference"])
async def infer(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Recebe os dados de um cliente e retorna a predição de risco de churn.

    Nome canônico do endpoint de inferência (`POST /infer`, alinhado à
    convenção de plataformas de model serving). Veja também `POST /predict`,
    mantido como alias.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível. Verifique se o treino foi executado ('make train').",
        )

    return _predict_and_track(payload)


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["inference"],
    summary="Alias de /infer (mantido por compatibilidade)",
)
async def predict(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Alias de `POST /infer`. Mantido para não quebrar integrações
    existentes que já usam o nome `/predict`.
    """
    return await infer(payload)


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
    for prediction in predictions:
        observe_prediction(prediction.risk_level)

    return BatchChurnPredictionResponse(predictions=predictions, count=len(predictions))
