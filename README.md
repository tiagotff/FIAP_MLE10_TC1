# Tech Challenge 1 — Previsão de Churn com Pipeline Profissional End-to-End

[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)]()
[![Etapa](https://img.shields.io/badge/etapa%20atual-1%20de%204-blue)]()

Projeto da Fase 1 da Pós Tech (FIAP) — **MLE10 / Grupo Tech Challenge 1**.

Construção, do zero, de um modelo de previsão de churn (cancelamento) de
clientes de telecomunicações, usando uma rede neural (MLP) em PyTorch,
comparada com baselines de Scikit-Learn, com rastreamento de experimentos via
MLflow e, nas etapas seguintes, servido via API FastAPI.

---

## Sumário

- [Contexto do problema](#contexto-do-problema)
- [Status do projeto](#status-do-projeto)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup do ambiente](#setup-do-ambiente)
- [Como executar](#como-executar)
- [Dataset](#dataset)
- [Resultados da Etapa 1](#resultados-da-etapa-1)
- [Próximas etapas](#próximas-etapas)

---

## Contexto do problema

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
A diretoria precisa de um modelo preditivo de churn que classifique clientes
com risco de cancelamento, permitindo ações de retenção direcionadas. O
projeto cobre todo o ciclo de vida de ML: entendimento do problema, EDA,
modelagem, engenharia de software e API de inferência, com boas práticas de
reprodutibilidade, testes e documentação.

## Status do projeto

| Etapa | Descrição | Status |
|---|---|---|
| **1** | Entendimento e Preparação (EDA, ML Canvas, Baselines, MLflow) | ✅ Concluída |
| **2** | Modelagem com Redes Neurais (MLP em PyTorch) | ⏳ Próxima |
| **3** | Engenharia e API (refatoração, FastAPI, testes) | ⏳ Pendente |
| **4** | Documentação e Entrega Final (Model Card, vídeo STAR) | ⏳ Pendente |

## Estrutura do repositório

```
.
├── data/
│   ├── raw/                 # Dataset bruto (versionado: é pequeno e público)
│   └── processed/            # Dados processados (não versionado)
├── docs/
│   └── ml_canvas.md          # ML Canvas do projeto (Etapa 1)
├── models/                   # Artefatos de modelo treinados (não versionado)
├── notebooks/
│   └── 01_eda_baselines.ipynb  # EDA completa + baselines (Etapa 1)
├── scripts/
│   └── generate_eda_notebook.py  # Script que gera o notebook de forma reprodutível
├── src/
│   └── churn_prediction/
│       ├── __init__.py
│       └── config.py         # Configuração central: seeds, paths, MLflow
├── tests/                    # Testes automatizados (a partir da Etapa 3)
├── .gitignore
├── pyproject.toml            # Single source of truth: deps, ruff, pytest
└── README.md
```

## Setup do ambiente

> ⚠️ **Use Python 3.10, 3.11 ou 3.12.** Versões mais novas (3.13/3.14) **não
> são suportadas** — o MLflow 3.x falha ao iniciar nessas versões
> (`ImportError: cannot import name 'Traversable' from 'importlib.abc'`),
> pois depende de uma API do `importlib.abc` removida no Python 3.13+. O
> `pyproject.toml` já restringe `requires-python` para evitar essa instalação,
> mas é importante **criar o ambiente virtual já com a versão certa** (veja o
> passo 2 abaixo).

Pré-requisitos: **Python 3.10, 3.11 ou 3.12** (recomendado: 3.12).

No Windows, se você tiver várias versões instaladas, confira quais estão
disponíveis com:

\`\`\`bash
py -0
\`\`\`

Se não tiver nenhuma versão entre 3.10 e 3.12, baixe o instalador do Python
3.12 em https://www.python.org/downloads/release/python-3127/ (marque
"Add python.exe to PATH" durante a instalação).

\`\`\`bash
# 1. Clonar o repositório
git clone https://github.com/tiagotff/FIAP_MLE10_TC1.git
cd FIAP_MLE10_TC1

# 2. Criar o ambiente virtual com uma versão suportada (ex.: 3.12)
python3.12 -m venv .venv          # Linux/macOS
# py -3.12 -m venv .venv          # Windows

# 3. Ativar o ambiente virtual
source .venv/bin/activate         # Linux/macOS
# source .venv/Scripts/activate   # Windows (Git Bash)
# .venv\Scripts\activate.bat      # Windows (cmd)
# .venv\Scripts\Activate.ps1      # Windows (PowerShell)

# 4. Instalar o projeto e suas dependências (modo editável, com extras de dev)
pip install -e ".[dev]"
\`\`\`

Isso instala todas as dependências declaradas no `pyproject.toml`: PyTorch,
Scikit-Learn, MLflow, FastAPI, Pydantic, Pandera, ferramentas de teste
(`pytest`) e lint (`ruff`).

## Como executar

### Notebook de EDA e baselines (Etapa 1)

```bash
jupyter notebook notebooks/01_eda_baselines.ipynb
```

O notebook já vem executado com os outputs salvos (gráficos, tabelas e
métricas reais), mas pode ser re-executado do início — basta rodar todas as
células (`Run All`). Ele:

1. Carrega e valida a qualidade do dataset bruto.
2. Realiza a EDA completa (distribuição do target, variáveis numéricas e
   categóricas, correlações).
3. Treina e avalia baselines (`DummyClassifier`, Regressão Logística) com
   validação cruzada estratificada.
4. Registra os experimentos no MLflow (backend SQLite local: `mlflow.db`).

Para regenerar o notebook do zero a partir do script-fonte (útil após editar
`scripts/generate_eda_notebook.py`):

```bash
python scripts/generate_eda_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_baselines.ipynb
```

### Visualizar experimentos no MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Depois acesse `http://localhost:5000` no navegador.

### Lint

```bash
ruff check .
```

## Troubleshooting

**`ImportError: cannot import name 'Traversable' from 'importlib.abc'` ao
rodar `mlflow ui` ou `mlflow.start_run()`.**
Você está usando Python 3.13 ou 3.14. Recrie o ambiente virtual com Python
3.10–3.12 (veja [Setup do ambiente](#setup-do-ambiente)).

**`[WinError 10013] Foi feita uma tentativa de acesso a um soquete...` ao
rodar `mlflow ui` no Windows.**
A porta padrão (5000) está bloqueada (uso por outro processo, antivírus ou
reserva do sistema). Use outra porta:

\`\`\`bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
\`\`\`

E acesse `http://localhost:5001`.


## Dataset

**Telco Customer Churn (IBM)** — dataset público de classificação binária,
com variáveis tabulares de clientes de uma operadora de telecomunicações
fictícia.

- **Fonte**: [IBM — telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Volume**: 7.043 registros, 20 features + variável-alvo (`Churn`)
- **Licença/uso**: dataset público, amplamente utilizado para fins educacionais.
- **Localização no repositório**: `data/raw/telco_customer_churn.csv`

## Resultados da Etapa 1

EDA completa documentada em [`notebooks/01_eda_baselines.ipynb`](notebooks/01_eda_baselines.ipynb).
Principais achados:

- Dataset sem `NaN` explícitos, mas com 11 registros de `TotalCharges` vazio
  (clientes com `tenure=0`) — tratado via imputação para `0`.
- Variável-alvo desbalanceada: **~73,5% "No churn"** vs. **~26,5% "Churn"**.
- Variáveis mais discriminantes: `Contract`, `tenure`, `InternetService`,
  `PaymentMethod`.

### Baselines (CV 5-fold estratificada, dados de treino)

| Modelo | AUC-ROC | PR-AUC | F1 | Acurácia |
|---|---|---|---|---|
| `DummyClassifier` (estratificado) | 0.507 | 0.269 | 0.276 | 0.614 |
| Regressão Logística (`class_weight=balanced`) | **0.846** | **0.660** | **0.629** | 0.749 |

A Regressão Logística supera amplamente o piso aleatório, confirmando sinal
preditivo real nos dados — pré-requisito para a Etapa 2 (rede neural).

ML Canvas completo (stakeholders, métricas de negócio, SLOs, riscos): ver
[`docs/ml_canvas.md`](docs/ml_canvas.md).

## Próximas etapas

- **Etapa 2**: construção e treinamento da MLP em PyTorch, com early stopping;
  comparação com os baselines usando ≥ 4 métricas; análise de trade-off de
  custo (falso positivo vs. falso negativo).
- **Etapa 3**: refatoração em módulos (`src/`), pipeline reprodutível, testes
  (pytest, pandera), API FastAPI (`/predict`, `/health`), logging estruturado.
- **Etapa 4**: Model Card, plano de monitoramento, vídeo STAR e
  (opcional) deploy em nuvem.

---

## Equipe

Grupo Tech Challenge 1 — FIAP MLE10.

Dúvidas? Procure o grupo no Discord do curso.
