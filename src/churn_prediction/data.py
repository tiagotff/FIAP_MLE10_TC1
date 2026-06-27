"""Carga e preparação dos dados de churn.

Centraliza a lógica de leitura do dataset bruto, tratamento de qualidade
(TotalCharges) e split treino/teste, para ser reutilizada tanto nos notebooks
quanto nos testes automatizados (Etapa 3) sem duplicar código.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_prediction.config import RAW_DATASET_PATH, SEED, TARGET_COLUMN, TEST_SIZE


def load_raw_dataset(path=RAW_DATASET_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto e aplica a correção de qualidade conhecida.

    Corrige `TotalCharges`: vem como string e tem 11 registros vazios
    (clientes com tenure=0, ainda sem cobrança), tratados como 0.0.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) e target binário (y) a partir do DataFrame bruto."""
    X = df.drop(columns=["customerID", TARGET_COLUMN])
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
):
    """Split treino/teste estratificado pelo target, com seed fixa."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna (features numéricas, features categóricas) do dataset de churn."""
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features
