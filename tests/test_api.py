"""Testes da API de inferência (FastAPI), usando TestClient (httpx).

Cobrem os contratos de todos os endpoints — `/`, `/health`, `/ready`,
`/metadata`, `/metrics`, `/infer`, `/predict` (alias) e `/predict/batch`:
códigos de status, formato da resposta, validação Pydantic de entrada,
headers de latência, CORS e o tratamento de erro genérico (que nunca deve
expor detalhes internos ao cliente).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api import app


@pytest.fixture
def client():
    """Cliente de teste que executa o ciclo de vida (lifespan) da aplicação,
    garantindo que o modelo seja carregado antes dos testes, como ocorreria
    em produção.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _skip_if_model_unavailable():
    """Pula os testes de /predict se os artefatos do modelo não existirem
    (ex.: ambiente limpo de CI sem o treino executado antes).
    """
    from churn_prediction.inference import MODEL_PATH, PREPROCESSOR_PATH

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Artefatos do modelo não encontrados — execute 'python -m churn_prediction.train' antes (com PYTHONPATH=src).")


def test_root_endpoint_returns_api_info(client):
    """GET / deve retornar informações básicas da API, evitando um 404
    "vazio" na rota raiz, com links para todos os endpoints canônicos.
    """
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Churn Prediction API"
    assert body["docs_url"] == "/docs"
    assert body["infer_url"] == "/infer"
    assert body["ready_url"] == "/ready"
    assert body["metadata_url"] == "/metadata"
    assert body["metrics_url"] == "/metrics"


def test_health_endpoint_is_simple_liveness_check(client):
    """GET /health é uma liveness probe simples — não depende do modelo
    estar carregado, apenas confirma que o processo da API está respondendo.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_reports_model_loaded(client):
    """GET /ready deve reportar 'ready' e model_loaded=True quando o modelo
    e o pipeline foram carregados com sucesso (pré-requisito: treino
    já executado — garantido pela fixture _skip_if_model_unavailable).
    """
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_metadata_endpoint_returns_model_info_and_schemas(client):
    """GET /metadata deve expor as métricas do modelo e os JSON Schemas de
    entrada/saída esperados por /infer — permitindo que um cliente
    descubra programaticamente o contrato da API.
    """
    response = client.get("/metadata")

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["model_info"]["test_roc_auc"] <= 1.0
    assert "properties" in body["input_schema"]
    assert "gender" in body["input_schema"]["properties"]
    assert "properties" in body["output_schema"]


def test_metrics_endpoint_returns_prometheus_format(client):
    """GET /metrics deve retornar métricas no formato texto do Prometheus,
    incluindo os contadores customizados desta API depois de pelo menos
    uma requisição ter sido processada.
    """
    client.get("/health")  # garante que existe ao menos uma métrica registrada

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "churn_api_requests_total" in body
    assert "churn_api_request_latency_seconds" in body


def test_infer_endpoint_returns_valid_response(client, valid_prediction_payload):
    """POST /infer, com um payload válido, deve retornar 200 e um corpo
    de resposta que respeita o schema ChurnPredictionResponse — este é o
    endpoint canônico de inferência.
    """
    response = client.post("/infer", json=valid_prediction_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["risk_level"] in {"low", "medium", "high"}
    assert "model_version" in body


def test_predict_alias_returns_same_result_as_infer(client, valid_prediction_payload):
    """POST /predict deve ser um alias funcionalmente idêntico a /infer,
    preservado por compatibilidade com integrações existentes.
    """
    infer_response = client.post("/infer", json=valid_prediction_payload).json()
    predict_response = client.post("/predict", json=valid_prediction_payload).json()

    assert infer_response["churn_probability"] == predict_response["churn_probability"]
    assert infer_response["risk_level"] == predict_response["risk_level"]


def test_predict_endpoint_returns_valid_response(client, valid_prediction_payload):
    """POST /predict, com um payload válido, deve retornar 200 e um corpo
    de resposta que respeita o schema ChurnPredictionResponse.
    """
    response = client.post("/predict", json=valid_prediction_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["risk_level"] in {"low", "medium", "high"}
    assert "model_version" in body


def test_predict_endpoint_rejects_invalid_category(client, valid_prediction_payload):
    """POST /predict com uma categoria fora do domínio esperado (ex.: um
    Contract inexistente) deve ser rejeitado com HTTP 422, antes de chegar
    à lógica de inferência.
    """
    invalid_payload = {**valid_prediction_payload, "Contract": "Three years"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_field(client, valid_prediction_payload):
    """POST /predict sem um campo obrigatório deve retornar HTTP 422."""
    incomplete_payload = {k: v for k, v in valid_prediction_payload.items() if k != "tenure"}

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422


def test_response_includes_latency_headers(client):
    """O middleware de latência deve adicionar os headers X-Request-ID e
    X-Process-Time-Ms em toda resposta, incluindo erros de validação.
    """
    response = client.get("/health")

    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers
    assert float(response.headers["x-process-time-ms"]) >= 0


def test_predict_low_risk_profile_returns_low_probability(client, valid_prediction_payload):
    """Um perfil de cliente fiel (tenure alto, contrato de 2 anos, pagamento
    automático) deve receber uma probabilidade de churn sensivelmente menor
    do que um perfil de risco (mês-a-mês, tenure baixo) — valida que a API
    está de fato usando o modelo treinado, não um valor fixo/aleatório.
    """
    low_risk_payload = {
        **valid_prediction_payload,
        "tenure": 70,
        "Contract": "Two year",
        "PaymentMethod": "Credit card (automatic)",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "MonthlyCharges": 20.0,
        "TotalCharges": 1400.0,
    }

    high_risk_response = client.post("/predict", json=valid_prediction_payload).json()
    low_risk_response = client.post("/predict", json=low_risk_payload).json()

    assert low_risk_response["churn_probability"] < high_risk_response["churn_probability"]


def test_predict_batch_endpoint_returns_one_prediction_per_customer(client, valid_prediction_payload):
    """POST /predict/batch deve retornar uma predição para cada cliente do
    lote, na mesma ordem em que foram enviados.
    """
    response = client.post(
        "/predict/batch", json={"customers": [valid_prediction_payload, valid_prediction_payload]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    # Mesmo payload enviado duas vezes deve produzir a mesma predição.
    assert body["predictions"][0]["churn_probability"] == body["predictions"][1]["churn_probability"]


def test_predict_batch_endpoint_rejects_empty_list(client):
    """POST /predict/batch com uma lista vazia de clientes deve ser
    rejeitado com HTTP 422 (min_length=1 no schema).
    """
    response = client.post("/predict/batch", json={"customers": []})

    assert response.status_code == 422


def test_predict_batch_endpoint_rejects_batch_above_limit(client, valid_prediction_payload):
    """POST /predict/batch com mais de 500 clientes deve ser rejeitado com
    HTTP 422, antes de qualquer processamento pelo modelo.
    """
    oversized_batch = {"customers": [valid_prediction_payload] * 501}

    response = client.post("/predict/batch", json=oversized_batch)

    assert response.status_code == 422


def test_cors_headers_present_on_preflight_request(client):
    """Uma requisição CORS preflight (OPTIONS) deve receber os headers
    Access-Control-* que habilitam consumo da API a partir de um navegador.
    """
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") == "http://example.com"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_unhandled_exception_returns_generic_500_without_internal_details(valid_prediction_payload):
    """Uma exceção inesperada na camada de inferência deve resultar em HTTP
    500 com uma mensagem genérica — nenhum detalhe interno (tipo da
    exceção, mensagem original, stack trace) deve ser exposto ao cliente.

    Testado via /predict, mas o handler genérico vale para qualquer rota
    (incluindo /infer, que internamente chama a mesma lógica de predição).

    Usa raise_server_exceptions=False neste teste específico: por padrão o
    TestClient re-levanta exceções não tratadas (útil para debugar testes),
    mas aqui queremos validar o comportamento real de produção, em que o
    exception_handler genérico converte a exceção em uma resposta HTTP 500.
    """
    from churn_prediction.inference import predictor

    original_predict = predictor.predict

    def _broken_predict(_payload):
        raise ValueError("detalhe interno sensível que não deve aparecer na resposta")

    predictor.predict = _broken_predict
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post("/predict", json=valid_prediction_payload)
    finally:
        predictor.predict = original_predict

    assert response.status_code == 500
    body = response.json()
    assert "sensível" not in body["detail"]
    assert "ValueError" not in body["detail"]
