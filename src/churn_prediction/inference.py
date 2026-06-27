"""Lógica de inferência: carrega os artefatos treinados e gera predições.

Separado da camada HTTP (api.py) por responsabilidade única — esta lógica
pode ser testada isoladamente (testes de schema/smoke) sem precisar
instanciar um servidor web.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import torch

from churn_prediction.config import MODELS_DIR
from churn_prediction.logging_config import get_logger
from churn_prediction.mlp_model import ChurnMLP, predict_proba
from churn_prediction.schemas import ChurnPredictionRequest, ChurnPredictionResponse

logger = get_logger(__name__)

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "mlp_model.pt"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RISK_THRESHOLDS = {"low": 0.3, "medium": 0.6}  # >= medium e < high vira "medium"; >= 0.6 vira "high"


class ModelNotLoadedError(RuntimeError):
    """Levantado quando uma predição é solicitada antes do modelo ser carregado."""


class ChurnPredictor:
    """Encapsula o modelo treinado (preprocessor + MLP) para uso em inferência.

    Carrega os artefatos uma única vez (no `load()`), mantendo-os em memória
    para predições subsequentes — evita o custo de I/O e desserialização a
    cada requisição da API.
    """

    def __init__(self):
        self.preprocessor = None
        self.model: ChurnMLP | None = None
        self.metadata: dict | None = None

    @property
    def is_loaded(self) -> bool:
        return self.preprocessor is not None and self.model is not None

    def load(self) -> None:
        """Carrega o preprocessor, o modelo PyTorch e os metadados do disco."""
        if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
            logger.error(
                "Artefatos do modelo não encontrados",
                extra={"preprocessor_path": str(PREPROCESSOR_PATH), "model_path": str(MODEL_PATH)},
            )
            raise FileNotFoundError(
                f"Artefatos não encontrados em {MODELS_DIR}. Execute "
                "'python -m churn_prediction.train' antes de iniciar a API."
            )

        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model = ChurnMLP(
            n_features=checkpoint["n_features"],
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model

        if METADATA_PATH.exists():
            with METADATA_PATH.open(encoding="utf-8") as f:
                self.metadata = json.load(f)

        logger.info(
            "Modelo carregado com sucesso",
            extra={"model_version": self.model_version},
        )

    @property
    def model_version(self) -> str:
        if self.metadata is not None:
            return self.metadata.get("model_version", "unknown")
        return "unknown"

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < RISK_THRESHOLDS["low"]:
            return "low"
        if probability < RISK_THRESHOLDS["medium"]:
            return "medium"
        return "high"

    def predict(self, request: ChurnPredictionRequest) -> ChurnPredictionResponse:
        """Gera uma predição de churn para um único cliente."""
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "O modelo ainda não foi carregado. Chame .load() antes de prever."
            )

        input_df = pd.DataFrame([request.model_dump()])
        X_processed = self.preprocessor.transform(input_df)

        probability = float(predict_proba(self.model, X_processed)[0])
        prediction = probability >= 0.5

        return ChurnPredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=self._risk_level(probability),
            model_version=self.model_version,
        )


# Instância única (singleton) reutilizada por toda a aplicação FastAPI.
predictor = ChurnPredictor()
