"""Treina o modelo final de produção (MLP) e salva os artefatos em models/.

Este script consolida a decisão tomada na Etapa 2 (a MLP foi escolhida como
modelo de produção, por ter o melhor recall e o melhor resultado no
framework de custo de negócio) e produz os artefatos necessários para a
API de inferência:

- `models/preprocessor.joblib`: pipeline de pré-processamento ajustado.
- `models/mlp_model.pt`: pesos treinados da MLP.
- `models/model_metadata.json`: metadados (métricas, parâmetros, versão).

Uso:
    python -m churn_prediction.train
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_prediction.business_cost import compute_business_cost
from churn_prediction.config import MODELS_DIR, SEED, set_global_seed
from churn_prediction.data import (
    load_raw_dataset,
    split_features_target,
    train_test_split_stratified,
)
from churn_prediction.logging_config import get_logger
from churn_prediction.mlp_model import ChurnMLP, predict_proba, train_mlp
from churn_prediction.pipeline import build_preprocessing_pipeline

logger = get_logger(__name__)

MODEL_VERSION = "1.0.0"
HIDDEN_SIZES = (64, 32)
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MAX_EPOCHS = 200


def train_and_save_model() -> dict:
    """Executa o pipeline completo de treino e salva os artefatos finais.

    Retorna um dicionário com as métricas de avaliação no holdout de teste,
    útil para inspeção programática (ex.: em testes automatizados).
    """
    set_global_seed(SEED)
    logger.info("Iniciando treino do modelo de produção", extra={"model_version": MODEL_VERSION})

    df = load_raw_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    X_tr, X_val, y_tr, y_val = train_test_split_stratified(X_train, y_train, test_size=0.2)

    logger.info(
        "Dados carregados e divididos",
        extra={
            "n_train": X_tr.shape[0],
            "n_val": X_val.shape[0],
            "n_test": X_test.shape[0],
        },
    )

    preprocessor = build_preprocessing_pipeline(X_tr)
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    n_features = X_tr_proc.shape[1]
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    logger.info(
        "Pré-processamento concluído",
        extra={"n_features": n_features, "pos_weight": round(float(pos_weight), 4)},
    )

    model = ChurnMLP(n_features=n_features, hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT)

    train_result = train_mlp(
        model,
        X_tr_proc,
        y_tr.to_numpy(),
        X_val_proc,
        y_val.to_numpy(),
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        pos_weight=pos_weight,
        seed=SEED,
        verbose=False,
    )

    logger.info(
        "Treinamento da MLP concluído",
        extra={
            "n_epochs_trained": train_result["n_epochs_trained"],
            "best_val_loss": round(train_result["best_val_loss"], 4),
        },
    )

    test_proba = predict_proba(model, X_test_proc)
    test_pred = (test_proba >= 0.5).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
        "f1": float(f1_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
    }

    business_metrics = compute_business_cost(y_test.to_numpy(), test_pred, X_test["MonthlyCharges"])

    logger.info("Avaliação no holdout de teste concluída", extra={"test_metrics": test_metrics})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    model_path = MODELS_DIR / "mlp_model.pt"
    metadata_path = MODELS_DIR / "model_metadata.json"

    joblib.dump(preprocessor, preprocessor_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden_sizes": HIDDEN_SIZES,
            "dropout": DROPOUT,
        },
        model_path,
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "seed": SEED,
        "architecture": {
            "hidden_sizes": list(HIDDEN_SIZES),
            "dropout": DROPOUT,
            "n_features": n_features,
        },
        "training": {
            "n_epochs_trained": train_result["n_epochs_trained"],
            "best_val_loss": round(train_result["best_val_loss"], 4),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "pos_weight": round(float(pos_weight), 4),
        },
        "test_metrics": test_metrics,
        "business_metrics": business_metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Artefatos salvos com sucesso",
        extra={
            "preprocessor_path": str(preprocessor_path),
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    )

    return metadata


if __name__ == "__main__":
    result_metadata = train_and_save_model()
    logger.info("Treino finalizado", extra={"summary": result_metadata["test_metrics"]})
