"""Testes do model_registry: comportamento local (no-op) e modo bucket
(download do Cloud Storage), este último usando mocks — não requer
credenciais reais do GCP nem acesso de rede.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from churn_prediction import model_registry


@pytest.fixture(autouse=True)
def _clean_model_bucket_env():
    """Garante que MODEL_BUCKET não 'escape' entre testes."""
    original = os.environ.pop(model_registry.MODEL_BUCKET_ENV_VAR, None)
    yield
    if original is not None:
        os.environ[model_registry.MODEL_BUCKET_ENV_VAR] = original
    else:
        os.environ.pop(model_registry.MODEL_BUCKET_ENV_VAR, None)


def test_ensure_model_artifacts_is_noop_without_bucket_configured():
    """Sem MODEL_BUCKET definido, a função não deve tentar nenhuma chamada
    de rede/GCS — comportamento padrão de desenvolvimento local.
    """
    with patch("google.cloud.storage.Client") as mock_client_cls:
        model_registry.ensure_model_artifacts()

    mock_client_cls.assert_not_called()


def test_ensure_model_artifacts_downloads_required_files_when_bucket_configured():
    """Com MODEL_BUCKET definido, deve baixar mlp_model.pt e
    preprocessor.joblib do bucket configurado.
    """
    os.environ[model_registry.MODEL_BUCKET_ENV_VAR] = "test-bucket"

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("google.cloud.storage.Client", return_value=mock_client):
        model_registry.ensure_model_artifacts()

    mock_client.bucket.assert_called_once_with("test-bucket")
    assert mock_blob.download_to_filename.call_count == len(model_registry.ARTIFACT_FILENAMES)


def test_ensure_model_artifacts_tolerates_missing_optional_metadata():
    """Se model_metadata.json não existir no bucket, a função não deve
    falhar — apenas os artefatos obrigatórios (.pt, .joblib) são críticos.
    """
    os.environ[model_registry.MODEL_BUCKET_ENV_VAR] = "test-bucket"

    def make_blob(exists: bool) -> MagicMock:
        blob = MagicMock()
        blob.exists.return_value = exists
        return blob

    blobs_by_suffix = {
        "mlp_model.pt": make_blob(exists=True),
        "preprocessor.joblib": make_blob(exists=True),
        "model_metadata.json": make_blob(exists=False),
    }

    mock_bucket = MagicMock()
    mock_bucket.blob.side_effect = lambda path: next(
        blob for suffix, blob in blobs_by_suffix.items() if path.endswith(suffix)
    )

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("google.cloud.storage.Client", return_value=mock_client):
        model_registry.ensure_model_artifacts()  # não deve lançar exceção

    assert blobs_by_suffix["mlp_model.pt"].download_to_filename.called
    assert blobs_by_suffix["preprocessor.joblib"].download_to_filename.called
    assert not blobs_by_suffix["model_metadata.json"].download_to_filename.called


def test_ensure_model_artifacts_respects_custom_prefix():
    """O prefixo do bucket deve ser configurável via MODEL_BUCKET_PREFIX,
    permitindo organizar múltiplas versões de modelo no mesmo bucket.
    """
    os.environ[model_registry.MODEL_BUCKET_ENV_VAR] = "test-bucket"
    os.environ[model_registry.MODEL_BUCKET_PREFIX_ENV_VAR] = "models/v2"

    mock_blob = MagicMock()
    mock_blob.exists.return_value = True

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    try:
        with patch("google.cloud.storage.Client", return_value=mock_client):
            model_registry.ensure_model_artifacts()

        called_paths = [call.args[0] for call in mock_bucket.blob.call_args_list]
        assert all(path.startswith("models/v2/") for path in called_paths)
    finally:
        os.environ.pop(model_registry.MODEL_BUCKET_PREFIX_ENV_VAR, None)
