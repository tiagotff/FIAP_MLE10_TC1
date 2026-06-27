"""Framework de custo de negócio para avaliação de modelos de churn.

Formaliza o framework descrito no ML Canvas (docs/ml_canvas.md): cada decisão
do modelo (TP, FP, FN, TN) tem um custo ou ganho associado, permitindo
comparar modelos por uma métrica de negócio, não apenas estatística.

Custos assumidos (parametrizáveis):
- Falso Negativo (FN): cliente que ia cancelar e não foi identificado.
  Custo = receita anual perdida, aproximada por 12x a cobrança mensal.
- Falso Positivo (FP): cliente sinalizado como risco, mas que não ia
  cancelar. Custo = custo fixo de uma campanha de retenção (contato, oferta).
- Verdadeiro Positivo (TP): cliente que ia cancelar e foi retido a tempo.
  Ganho = receita anual preservada, líquida do custo da campanha.
- Verdadeiro Negativo (TN): nenhuma ação, sem custo nem ganho.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CAMPAIGN_COST: float = 50.0  # custo fixo estimado de uma campanha de retenção (R$)
DEFAULT_MONTHS_RETAINED: int = 12  # meses de receita considerados na perda/ganho


def estimate_customer_value(monthly_charges: pd.Series, months: int = DEFAULT_MONTHS_RETAINED) -> pd.Series:
    """Estima o valor anualizado de um cliente a partir da cobrança mensal."""
    return monthly_charges * months


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    monthly_charges: pd.Series,
    campaign_cost: float = DEFAULT_CAMPAIGN_COST,
    months_retained: int = DEFAULT_MONTHS_RETAINED,
) -> dict:
    """Calcula o custo líquido total de um conjunto de previsões.

    Retorna um dicionário com a contagem de cada tipo de decisão e o custo
    líquido total (quanto menor/mais negativo, melhor para o negócio — custo
    negativo aqui significa ganho líquido).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    customer_value = estimate_customer_value(monthly_charges, months_retained).to_numpy()

    is_tp = (y_true == 1) & (y_pred == 1)
    is_fp = (y_true == 0) & (y_pred == 1)
    is_fn = (y_true == 1) & (y_pred == 0)
    is_tn = (y_true == 0) & (y_pred == 0)

    # FN: perde o valor do cliente que cancelou sem ser identificado.
    cost_fn = customer_value[is_fn].sum()

    # FP: custo fixo da campanha de retenção, desnecessária nesse caso.
    cost_fp = is_fp.sum() * campaign_cost

    # TP: ganho líquido = valor do cliente retido menos o custo da campanha
    # que viabilizou a retenção. Representado como custo negativo (= ganho).
    gain_tp = customer_value[is_tp].sum() - is_tp.sum() * campaign_cost

    net_cost = cost_fn + cost_fp - gain_tp

    return {
        "n_tp": int(is_tp.sum()),
        "n_fp": int(is_fp.sum()),
        "n_fn": int(is_fn.sum()),
        "n_tn": int(is_tn.sum()),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "gain_tp": float(gain_tp),
        "net_cost": float(net_cost),
    }


def cost_summary_table(results: dict[str, dict]) -> pd.DataFrame:
    """Monta uma tabela comparativa de custo de negócio entre modelos.

    `results` é um dicionário {nome_do_modelo: saida_de_compute_business_cost}.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({"modelo": model_name, **metrics})
    return pd.DataFrame(rows).set_index("modelo")
