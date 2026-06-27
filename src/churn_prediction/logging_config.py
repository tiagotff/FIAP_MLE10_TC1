"""Configuração de logging estruturado para o projeto.

Boa prática obrigatória do Tech Challenge: nenhum módulo de produção usa
`print()`. Todo log é estruturado (JSON), facilitando ingestão por
ferramentas de observabilidade (ex.: ELK, Datadog, CloudWatch Logs) em um
cenário real de produção.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    """Formata cada registro de log como uma linha JSON.

    Inclui timestamp (UTC, ISO 8601), nível, logger de origem, mensagem e
    quaisquer campos extras passados via `extra=` na chamada de log (ex.:
    `logger.info("...", extra={"request_id": "abc"})`).
    """

    # Atributos padrão do LogRecord que não devem ser tratados como "extra".
    _RESERVED_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra_fields = {
            key: value for key, value in record.__dict__.items() if key not in self._RESERVED_KEYS
        }
        payload.update(extra_fields)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configura o logging raiz da aplicação para emitir JSON em stdout.

    Idempotente: chamadas repetidas não duplicam handlers.
    """
    root_logger = logging.getLogger()

    if any(isinstance(h.formatter, JSONFormatter) for h in root_logger.handlers):
        return  # já configurado

    root_logger.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado para o módulo chamador.

    Uso recomendado: `logger = get_logger(__name__)` no topo de cada módulo.
    """
    configure_logging()
    return logging.getLogger(name)
