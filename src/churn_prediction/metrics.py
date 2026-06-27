"""Métricas operacionais da API, expostas no formato Prometheus via `/metrics`.

Diferença importante em relação a `ModelInfo` (`schemas.py`): este módulo
mede a **operação da API** (quantas requisições, latência, taxa de erro) —
não a qualidade do modelo de ML. As duas coisas são observabilidade de
camadas diferentes e por isso ficam em endpoints diferentes (`/metrics` vs.
`/metadata`).
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Contador de requisições, particionado por rota, método e código de status
# — permite consultas como "taxa de erro do /infer nas últimas 5 min" no
# Prometheus/Grafana.
REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total de requisições HTTP recebidas pela API",
    labelnames=["method", "path", "status_code"],
)

# Histograma de latência por rota — permite calcular percentis (p50, p95,
# p99) no Prometheus, em vez de só uma média simples.
REQUEST_LATENCY_SECONDS = Histogram(
    "churn_api_request_latency_seconds",
    "Latência das requisições HTTP, em segundos",
    labelnames=["method", "path"],
)

# Contador específico de predições, particionado pelo nível de risco
# retornado — útil para acompanhar a distribuição de risco da carteira de
# clientes avaliada ao longo do tempo (ex.: detectar um desvio repentino).
PREDICTION_COUNT = Counter(
    "churn_predictions_total",
    "Total de predições de churn realizadas",
    labelnames=["risk_level"],
)


def observe_request(method: str, path: str, status_code: int, latency_seconds: float) -> None:
    """Registra uma requisição concluída nas métricas de contagem e latência."""
    REQUEST_COUNT.labels(method=method, path=path, status_code=str(status_code)).inc()
    REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(latency_seconds)


def observe_prediction(risk_level: str) -> None:
    """Registra uma predição realizada, por nível de risco."""
    PREDICTION_COUNT.labels(risk_level=risk_level).inc()


def render_latest_metrics() -> tuple[bytes, str]:
    """Gera o corpo da resposta de `/metrics` no formato texto do Prometheus.

    Retorna (corpo_em_bytes, content_type) — o content_type correto é
    exigido pelo Prometheus para conseguir fazer o parsing (`scrape`).
    """
    return generate_latest(), CONTENT_TYPE_LATEST
