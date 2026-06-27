"""Configurações centrais do projeto: paths, seeds e parâmetros globais.

Este módulo é a fonte única de verdade para constantes usadas em todas as
etapas do pipeline (EDA, baselines, treinamento da MLP, API), garantindo
reprodutibilidade.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Seed global de reprodutibilidade (boa prática obrigatória do Tech Challenge)
# --------------------------------------------------------------------------
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fixa a seed em todas as bibliotecas relevantes (random, numpy, torch).

    O import de torch é feito de forma lazy (dentro da função) para que este
    módulo de configuração não exija PyTorch instalado quando usado apenas
    em etapas que não dependem dele (ex.: EDA pura com pandas).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Garante determinismo nas operações de cuDNN, quando disponível.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch ainda não é necessário nesta etapa (ex.: apenas EDA).
        pass


# --------------------------------------------------------------------------
# Paths do projeto
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_DATASET_PATH: Path = RAW_DATA_DIR / "telco_customer_churn.csv"

# --------------------------------------------------------------------------
# Parâmetros de dados
# --------------------------------------------------------------------------
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

# Proporção de holdout para teste final (usada nas etapas de modelagem)
TEST_SIZE: float = 0.2
N_SPLITS_CV: int = 5  # validação cruzada estratificada (k=5)

# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
# Observação: o MLflow 3.x colocou o backend de arquivo puro ("file:./mlruns")
# em modo de manutenção, recomendando um backend de banco de dados. Usamos
# SQLite local — mantém o tracking 100% local/sem dependências externas, mas
# já na forma suportada pelas versões atuais do MLflow.
MLFLOW_EXPERIMENT_NAME: str = "churn-prediction"
MLFLOW_TRACKING_URI: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
