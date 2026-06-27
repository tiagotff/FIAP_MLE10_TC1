"""Testes da API de inferência (FastAPI), usando TestClient (httpx).

Cobrem os contratos dos endpoints `/health` e `/predict`: códigos de status,
formato da resposta, validação Pydantic de entrada e o header de latência
inserido pelo middleware.
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
    (ex.: ambiente limpo de CI sem 'make train' executado antes).
    """
    from churn_prediction.inference import MODEL_PATH, PREPROCESSOR_PATH

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        pytest.skip("Artefatos do modelo não encontrados — execute 'make train' antes.")


def test_health_endpoint_returns_ok_when_model_loaded(client):
    """GET /health deve retornar 200 e reportar o modelo como carregado,
    assumindo que os artefatos já foram treinados.
    """
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] is not None


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
