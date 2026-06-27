"""Schemas Pydantic para validação de entrada/saída da API de inferência.

Os tipos `Literal` espelham exatamente os domínios observados na EDA (Etapa
1) para cada coluna categórica do dataset Telco Customer Churn — qualquer
valor fora desses domínios é rejeitado automaticamente pelo FastAPI com
HTTP 422, antes mesmo de chegar à lógica de predição.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    """Dados de um cliente, no mesmo formato do dataset de treino."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Meses de relacionamento com a operadora")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=1000, description="Cobrança mensal em moeda local")
    TotalCharges: float = Field(ge=0, le=100_000, description="Cobrança total acumulada")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class ChurnPredictionResponse(BaseModel):
    """Resultado da predição de churn para um cliente."""

    churn_probability: float = Field(ge=0, le=1, description="Probabilidade estimada de churn")
    churn_prediction: bool = Field(description="Predição binária (threshold=0.5)")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Faixa de risco derivada da probabilidade, para uso operacional"
    )
    model_version: str = Field(description="Versão do modelo que gerou esta predição")


class BatchChurnPredictionRequest(BaseModel):
    """Lote de clientes para predição em uma única chamada.

    Útil para o time de Retenção/CRM pontuar uma carteira de clientes de
    uma vez (ex.: rotina batch diária), em vez de uma chamada HTTP por
    cliente.
    """

    customers: list[ChurnPredictionRequest] = Field(
        min_length=1,
        max_length=500,
        description="Lista de clientes a serem avaliados (máximo 500 por requisição)",
    )


class BatchChurnPredictionResponse(BaseModel):
    """Resultado da predição em lote."""

    predictions: list[ChurnPredictionResponse]
    count: int = Field(description="Número de clientes avaliados nesta resposta")


class ModelInfo(BaseModel):
    """Resumo das métricas e metadados do modelo carregado, exposto via /health."""

    model_version: str
    trained_at: str | None = None
    test_roc_auc: float | None = None
    test_recall: float | None = None
    business_net_cost: float | None = None


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    model_info: ModelInfo | None = Field(
        default=None, description="Métricas resumidas do modelo, quando disponíveis"
    )


class RootResponse(BaseModel):
    """Resposta do endpoint raiz (/), com informações básicas da API."""

    name: str
    version: str
    docs_url: str
    health_url: str
    predict_url: str
