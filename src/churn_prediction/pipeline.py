"""Pipeline de pré-processamento reprodutível, com transformador customizado.

Encapsula toda a lógica de transformação de features em um único objeto
sklearn (`Pipeline`), reaproveitável entre treino, API de inferência e
testes — garantindo que a mesma transformação seja aplicada de forma
idêntica em todos os contextos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.data import get_feature_groups


class TenureBucketizer(BaseEstimator, TransformerMixin):
    """Transformador customizado: deriva uma feature categórica de faixa de tenure.

    Achado da EDA (Etapa 1): a taxa de churn cai de forma quase monotônica
    com o tempo de relacionamento (tenure). Discretizar em faixas captura
    esse efeito de forma mais explícita para os modelos lineares, além de
    servir como exemplo de transformador sklearn customizado exigido pelo
    desafio (transformadores custom no pipeline).

    Segue a API padrão do Scikit-Learn (`fit`/`transform`), podendo ser
    inserido em qualquer `Pipeline` ou `ColumnTransformer`.
    """

    def __init__(self, tenure_column: str = "tenure"):
        self.tenure_column = tenure_column
        self.bins_ = [-1, 12, 24, 48, 72, np.inf]
        self.labels_ = ["0-12", "13-24", "25-48", "49-72", "73+"]

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ARG002 - assinatura padrão sklearn
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["tenure_bucket"] = pd.cut(
            X[self.tenure_column], bins=self.bins_, labels=self.labels_
        ).astype(str)
        return X

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        return np.array(["tenure_bucket"])


def build_preprocessing_pipeline(X_reference: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline completo de pré-processamento.

    Etapas:
    1. `TenureBucketizer` (transformador custom): adiciona a feature derivada
       `tenure_bucket` ao DataFrame.
    2. `ColumnTransformer`: escala features numéricas e aplica one-hot às
       categóricas (incluindo a nova `tenure_bucket`).

    `X_reference` é usado apenas para detectar dinamicamente os grupos de
    features (numéricas vs. categóricas) a partir das colunas originais.
    """
    numeric_features, categorical_features = get_feature_groups(X_reference)
    categorical_features_with_bucket = [*categorical_features, "tenure_bucket"]

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_features_with_bucket,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("tenure_bucketizer", TenureBucketizer()),
            ("column_transformer", column_transformer),
        ]
    )
