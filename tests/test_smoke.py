"""Smoke tests: validam que os fluxos principais executam de ponta a ponta
sem lançar exceções, com tempo de execução reduzido (poucas épocas).

Smoke tests não verificam a qualidade do modelo (isso é papel dos notebooks
de avaliação nas Etapas 1/2) — apenas que a engenharia (pipeline, treino,
inferência) está íntegra e executável após qualquer alteração de código.
"""

from __future__ import annotations

from churn_prediction.config import SEED, set_global_seed
from churn_prediction.data import (
    load_raw_dataset,
    split_features_target,
    train_test_split_stratified,
)
from churn_prediction.inference import ChurnPredictor
from churn_prediction.mlp_model import ChurnMLP, predict_proba, train_mlp
from churn_prediction.pipeline import build_preprocessing_pipeline
from churn_prediction.schemas import ChurnPredictionRequest


def test_data_loading_smoke():
    """O dataset bruto deve carregar e produzir X/y não vazios."""
    df = load_raw_dataset()
    X, y = split_features_target(df)

    assert len(df) > 0
    assert len(X) == len(y)
    assert X.shape[1] > 0


def test_preprocessing_pipeline_smoke():
    """O pipeline de pré-processamento deve ajustar e transformar sem erros,
    produzindo uma matriz numérica sem valores nulos.
    """
    set_global_seed(SEED)
    df = load_raw_dataset()
    X, y = split_features_target(df)
    X_train, X_test, _, _ = train_test_split_stratified(X, y)

    pipeline = build_preprocessing_pipeline(X_train)
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    assert X_train_proc.shape[0] == len(X_train)
    assert X_test_proc.shape[1] == X_train_proc.shape[1]
    assert not pd_isna_any(X_train_proc)


def test_mlp_training_smoke():
    """Um treino reduzido (poucas épocas) da MLP deve executar sem lançar
    exceções e produzir probabilidades válidas (entre 0 e 1).
    """
    set_global_seed(SEED)
    df = load_raw_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    X_tr, X_val, y_tr, y_val = train_test_split_stratified(X_train, y_train, test_size=0.2)

    pipeline = build_preprocessing_pipeline(X_tr)
    X_tr_proc = pipeline.fit_transform(X_tr)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    model = ChurnMLP(n_features=X_tr_proc.shape[1], hidden_sizes=(8, 4), dropout=0.1)

    result = train_mlp(
        model,
        X_tr_proc,
        y_tr.to_numpy(),
        X_val_proc,
        y_val.to_numpy(),
        max_epochs=3,  # smoke test: só verifica que o loop executa, não converge
        batch_size=128,
        patience=3,
        seed=SEED,
    )

    probabilities = predict_proba(model, X_test_proc)

    assert result["n_epochs_trained"] <= 3
    assert probabilities.shape[0] == len(X_test)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


def test_inference_predictor_smoke(valid_prediction_payload):
    """O ChurnPredictor deve carregar os artefatos treinados (gerados por
    'python -m churn_prediction.train') e produzir uma predição válida.

    Pré-requisito: os artefatos em models/ devem existir. Caso não existam
    (ex.: ambiente de CI sem treino prévio), o teste é pulado.
    """
    predictor = ChurnPredictor()
    try:
        predictor.load()
    except FileNotFoundError:
        import pytest

        pytest.skip("Artefatos do modelo não encontrados — execute 'python -m churn_prediction.train' antes (com PYTHONPATH=src).")

    request = ChurnPredictionRequest.model_validate(valid_prediction_payload)
    response = predictor.predict(request)

    assert 0.0 <= response.churn_probability <= 1.0
    assert isinstance(response.churn_prediction, bool)
    assert response.risk_level in {"low", "medium", "high"}


def pd_isna_any(array) -> bool:
    """Helper local: verifica se uma matriz numpy contém algum NaN."""
    import numpy as np

    return bool(np.isnan(array).any())
