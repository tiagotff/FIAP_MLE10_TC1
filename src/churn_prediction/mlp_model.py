"""Rede neural MLP (Multi-Layer Perceptron) em PyTorch para previsão de churn.

Define a arquitetura do modelo e o loop de treinamento com early stopping e
mini-batching, seguindo as boas práticas exigidas pelo Tech Challenge.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    """MLP simples para classificação binária de churn.

    Arquitetura: camadas densas com ativação ReLU e Dropout para
    regularização, finalizando em uma única unidade de saída (logit) — a
    função de perda `BCEWithLogitsLoss` aplica a sigmoid internamente, o que
    é numericamente mais estável do que aplicar sigmoid manualmente.
    """

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (64, 32), dropout: float = 0.3):
        super().__init__()

        layers: list[nn.Module] = []
        in_features = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 1))  # logit de saída (sem sigmoid)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class EarlyStopping:
    """Interrompe o treinamento quando a métrica de validação para de melhorar.

    Monitora a perda de validação (menor é melhor) e restaura os pesos do
    melhor ponto observado ao final do treinamento.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state: dict | None = None
        self.counter = 0
        self.stopped_epoch: int | None = None

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Avalia se deve parar. Retorna True se o treino deve ser interrompido."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    pos_weight: float | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Treina a MLP com mini-batching e early stopping baseado na perda de validação.

    Retorna um dicionário com o histórico de treino (`train_loss`, `val_loss`
    por época) e o número de épocas efetivamente treinadas.
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    pos_weight_tensor = (
        torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight is not None else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        epoch_train_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f",
                epoch,
                epoch_train_loss,
                val_loss,
            )

        if early_stopping.step(val_loss, model, epoch):
            if verbose:
                logger.info("Early stopping na época %d (melhor val_loss=%.4f)", epoch, early_stopping.best_loss)
            break

    early_stopping.restore_best(model)

    return {
        "history": history,
        "n_epochs_trained": len(history["train_loss"]),
        "stopped_epoch": early_stopping.stopped_epoch,
        "best_val_loss": early_stopping.best_loss,
    }


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Retorna probabilidades (sigmoid dos logits) para um array de features."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_t)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()
