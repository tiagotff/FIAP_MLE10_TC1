"""Testes do app Streamlit (app/streamlit_app.py).

Cobrem apenas a lógica que não depende de renderização de UI (que exigiria
um framework de teste de UI dedicado, fora do escopo desta suíte):
resolução da URL da API e o wrapper de chamadas HTTP (`call_api`), usando
`responses` para simular a API sem precisar de um servidor real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import responses

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit_app as app  # noqa: E402 - import após ajuste de sys.path


@pytest.fixture(autouse=True)
def _reset_api_url_env(monkeypatch):
    """Garante uma URL de API conhecida e isolada em cada teste."""
    monkeypatch.setenv("CHURN_API_URL", "http://test-api.local")


def test_get_api_url_reads_from_env_var():
    assert app.get_api_url() == "http://test-api.local"


def test_get_api_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("CHURN_API_URL", "http://test-api.local/")
    assert app.get_api_url() == "http://test-api.local"


def test_get_api_url_falls_back_to_default_when_unset(monkeypatch):
    monkeypatch.delenv("CHURN_API_URL", raising=False)
    assert app.get_api_url() == app.DEFAULT_API_URL


@responses.activate
def test_call_api_returns_success_tuple_on_200():
    responses.add(
        responses.GET,
        "http://test-api.local/ready",
        json={"status": "ready", "model_loaded": True},
        status=200,
    )

    ok, data = app.call_api("GET", "/ready")

    assert ok is True
    assert data == {"status": "ready", "model_loaded": True}


@responses.activate
def test_call_api_returns_failure_tuple_with_detail_on_error_status():
    responses.add(
        responses.POST,
        "http://test-api.local/infer",
        json={"detail": "Modelo indisponível."},
        status=503,
    )

    ok, message = app.call_api("POST", "/infer", json={})

    assert ok is False
    assert "503" in message
    assert "Modelo indisponível." in message


@responses.activate
def test_call_api_handles_connection_error_gracefully():
    import requests as requests_lib

    responses.add(
        responses.GET,
        "http://test-api.local/ready",
        body=requests_lib.exceptions.ConnectionError("connection refused"),
    )

    ok, message = app.call_api("GET", "/ready")

    assert ok is False
    assert "Não foi possível conectar" in message


def test_render_risk_badge_includes_emoji_and_uppercase_label():
    assert app.render_risk_badge("high") == "🔴 HIGH"
    assert app.render_risk_badge("low") == "🟢 LOW"


def test_required_csv_columns_match_api_request_schema():
    """Garante que a lista de colunas exigidas no CSV do app continua
    sincronizada com os campos de ChurnPredictionRequest da API — evita que
    uma mudança no schema da API "quebre" silenciosamente o app sem que
    os testes sinalizem.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from churn_prediction.schemas import ChurnPredictionRequest

    api_fields = set(ChurnPredictionRequest.model_fields.keys())
    app_fields = set(app.REQUIRED_CSV_COLUMNS)

    assert app_fields == api_fields
