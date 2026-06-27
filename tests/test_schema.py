"""Testes de schema (pandera) para o dataset de churn.

Validam que os dados — tanto o dataset bruto real quanto dados sintéticos
no mesmo formato — respeitam o contrato de schema esperado pelo pipeline:
tipos corretos, domínios de valores categóricos, e ausência de nulos onde
não esperado. Isso funciona como uma rede de segurança contra mudanças
inesperadas na fonte de dados (ex.: uma nova categoria aparecendo, ou um
tipo de coluna mudando silenciosamente).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

from churn_prediction.config import RAW_DATASET_PATH
from churn_prediction.data import load_raw_dataset, split_features_target

# Schema do dataset bruto (antes do tratamento de TotalCharges) — reflete
# exatamente o que é observado na fonte (Etapa 1: TotalCharges como string).
RAW_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str, unique=True),
        "gender": Column(str, Check.isin(["Female", "Male"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.in_range(0, 120)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str, Check.isin(["Yes", "No", "No internet service"])),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, Check.in_range(0, 1000)),
        # TotalCharges é string no dataset bruto (achado de qualidade da Etapa 1).
        "TotalCharges": Column(str),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    strict=False,  # permite colunas adicionais (ex.: demais flags de serviço)
)

# Schema pós-tratamento (após load_raw_dataset corrigir TotalCharges).
PROCESSED_SCHEMA = DataFrameSchema(
    {
        "TotalCharges": Column(float, Check.in_range(0, 100_000), nullable=False),
        "tenure": Column(int, Check.in_range(0, 120)),
        "MonthlyCharges": Column(float, Check.in_range(0, 1000)),
    },
    strict=False,
)


def test_raw_dataset_matches_expected_schema():
    """O CSV bruto, lido diretamente, deve respeitar o schema documentado."""
    df = pd.read_csv(RAW_DATASET_PATH)
    RAW_SCHEMA.validate(df)


def test_processed_dataset_has_no_invalid_total_charges():
    """Após o tratamento de qualidade (Etapa 1), TotalCharges deve ser numérico,
    sem nulos e dentro de uma faixa plausível — não deve restar nenhuma string
    vazia ou valor fora do domínio esperado.
    """
    df = load_raw_dataset()
    PROCESSED_SCHEMA.validate(df)
    assert df["TotalCharges"].dtype == float
    assert df["TotalCharges"].isnull().sum() == 0


def test_synthetic_sample_matches_raw_schema(sample_raw_dataframe):
    """Dados sintéticos de teste (fixture) também devem aderir ao schema bruto,
    garantindo que os fixtures de teste continuem representativos do mundo real.
    """
    RAW_SCHEMA.validate(sample_raw_dataframe)


def test_split_features_target_preserves_row_count():
    """split_features_target não deve alterar o número de linhas nem
    introduzir/perder registros.
    """
    df = load_raw_dataset()
    X, y = split_features_target(df)
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert set(y.unique()).issubset({0, 1})


def test_invalid_category_is_rejected_by_schema():
    """Uma categoria fora do domínio conhecido (ex.: Contract inválido) deve
    ser rejeitada pela validação de schema, simulando um problema de dados
    upstream sendo capturado antes de chegar ao modelo.
    """
    df = pd.read_csv(RAW_DATASET_PATH).head(5).copy()
    df.loc[0, "Contract"] = "Three years"  # categoria inexistente no domínio

    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)
