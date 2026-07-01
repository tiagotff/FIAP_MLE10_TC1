# Tech Challenge 1 — Previsão de Churn com Pipeline Profissional End-to-End

[![Status](https://img.shields.io/badge/status-conclu%C3%ADdo-brightgreen)]()
[![Etapa](https://img.shields.io/badge/etapa%20atual-4%20de%204-blue)]()

Projeto da Fase 1 da Pós Tech (FIAP) — **MLE10 / Grupo Tech Challenge 1**.

Construção, do zero, de um modelo de previsão de churn (cancelamento) de
clientes de telecomunicações, usando uma rede neural (MLP) em PyTorch,
comparada com baselines de Scikit-Learn, com rastreamento de experimentos via
MLflow e, nas etapas seguintes, servido via API FastAPI.

---

## Quickstart

Sequência única, do zero, para quem acabou de clonar o repositório e ainda
não tem nenhum terminal aberto. Cada comando assume que você está na pasta
raiz do projeto (`cd FIAP_MLE10_TC1`) e usa **Git Bash** (Linux/macOS:
mesmos comandos; Windows PowerShell/cmd: ver variações na
[Setup do ambiente](#setup-do-ambiente) e em cada seção linkada abaixo).

> Você vai precisar de **2 terminais abertos ao mesmo tempo** mais adiante
> (um para a API, outro para o dashboard) — abra o segundo terminal só
> quando o passo indicar.

**Terminal 1** — do clone até a API rodando:

```bash
# 1. Clonar e entrar na pasta
git clone https://github.com/tiagotff/FIAP_MLE10_TC1.git
cd FIAP_MLE10_TC1

# 2. Criar e ativar o ambiente virtual (Python 3.10-3.12 — ver nota abaixo)
py -3.12 -m venv .venv
source .venv/Scripts/activate

# 3. Instalar o projeto e todas as dependências
pip install -e ".[dev]"

# 4. Treinar o modelo (gera os artefatos em models/, leva poucos segundos)
PYTHONPATH=src python -m churn_prediction.train

# 5. (Opcional, mas recomendado) Confirmar que tudo está correto: 38 testes devem passar
PYTHONPATH=src python -m pytest tests/ -v

# 6. Subir a API — deixe este terminal aberto e rodando
PYTHONPATH=src python -m uvicorn churn_prediction.api:app --host 127.0.0.1 --port 8000
```

Espere aparecer a linha `Uvicorn running on http://127.0.0.1:8000` antes de
seguir — **não digite mais nada neste terminal**, ele precisa continuar
rodando.

**Terminal 2** — novo terminal, com a API do Terminal 1 ainda rodando:

```bash
# 1. Entrar na pasta e ativar o mesmo ambiente virtual
cd FIAP_MLE10_TC1
source .venv/Scripts/activate

# 2. Confirmar que a API está respondendo (deve retornar model_loaded: true)
curl http://127.0.0.1:8000/ready

# 3. Subir o dashboard, apontando para a API do Terminal 1
CHURN_API_URL=http://127.0.0.1:8000 streamlit run app/streamlit_app.py
```

Abra `http://localhost:8501` no navegador — a sidebar deve mostrar
"✅ Modelo carregado e pronto" e você já pode usar as duas abas do
dashboard (avaliar um cliente, ou pontuar uma carteira via CSV).

> ⚠️ Se você não tiver Python 3.10-3.12 instalado, ou se `py -3.12` não
> for reconhecido, veja [Setup do ambiente](#setup-do-ambiente) — instalar
> a versão certa do Python é o único pré-requisito que pode exigir um
> passo extra antes de começar.
>
> Quer testar sem subir nada localmente? Aponte o Terminal 2 direto para a
> API já em produção (ver [Deploy em nuvem](#deploy-em-nuvem-bônus)):
> `CHURN_API_URL=https://churn-api-855490327597.us-central1.run.app streamlit run app/streamlit_app.py`
> — nesse caso, nem precisa do Terminal 1.

---

## Sumário

- [Quickstart](#quickstart)
- [Contexto do problema](#contexto-do-problema)
- [Status do projeto](#status-do-projeto)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup do ambiente](#setup-do-ambiente)
- [Como executar](#como-executar)
- [Troubleshooting](#troubleshooting)
- [API de inferência](#api-de-inferência)
- [Dashboard Streamlit](#dashboard-streamlit)
- [Testes automatizados](#testes-automatizados)
- [Dataset](#dataset)
- [Resultados da Etapa 1](#resultados-da-etapa-1)
- [Resultados da Etapa 2](#resultados-da-etapa-2)
- [Resultados da Etapa 3](#resultados-da-etapa-3)
- [Resultados da Etapa 4](#resultados-da-etapa-4)
- [Deploy em nuvem (bônus)](#deploy-em-nuvem-bônus)
- [Entrega final](#entrega-final)
- [Licença](#licença)

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
| **2** | Modelagem com Redes Neurais (MLP em PyTorch) | ✅ Concluída |
| **3** | Engenharia e API (refatoração, FastAPI, testes) | ✅ Concluída |
| **4** | Documentação e Entrega Final (Model Card, vídeo STAR) | ✅ Concluída |

## Estrutura do repositório

```
.
├── app/
│   └── streamlit_app.py         # Dashboard Streamlit (cliente visual da API)
├── data/
│   ├── raw/                 # Dataset bruto (versionado: é pequeno e público)
│   └── processed/            # Dados processados (não versionado)
├── docs/
│   ├── tech-challenge-fiap.pdf  # Enunciado oficial do desafio (material FIAP — ver Licença)
│   ├── ml_canvas.md             # ML Canvas do projeto (Etapa 1, atualizado nas Etapas 2/3)
│   ├── model_card.md            # Model Card completo (Etapa 4)
│   ├── deployment_architecture.md  # Arquitetura de deploy: batch + real-time (Etapa 4)
│   ├── monitoring_plan.md       # Plano de monitoramento: métricas, alertas, playbook (Etapa 4)
│   ├── Apresentacao_Tech_Challenge_1_MLE10_Churn.pptx  # Slides STAR (Etapa 4)
│   ├── Apresentacao_Tech_Challenge_1_MLE10_Churn.pdf   # Slides STAR, versão PDF
│   └── video_apresentacao_STAR_TC1_MLE.mp4             # Vídeo de apresentação (5 min, STAR)
├── models/                   # Artefatos de modelo treinados (não versionado)
├── notebooks/
│   ├── 01_eda_baselines.ipynb       # EDA completa + baselines (Etapa 1)
│   └── 02_mlp_model_comparison.ipynb  # MLP, baseline de árvore e comparação (Etapa 2)
├── scripts/
│   ├── generate_eda_notebook.py     # Gera o notebook da Etapa 1
│   ├── generate_mlp_notebook.py     # Gera o notebook da Etapa 2
│   ├── upload_model_to_gcs.sh       # Sobe os artefatos do modelo para o bucket GCS
│   └── deploy_streamlit_to_cloud_run.sh  # Build + deploy do dashboard no Cloud Run
├── src/
│   └── churn_prediction/
│       ├── __init__.py
│       ├── config.py           # Configuração central: seeds, paths, MLflow
│       ├── data.py             # Carga, tratamento de qualidade e splits
│       ├── pipeline.py         # Pipeline sklearn + transformador custom (TenureBucketizer)
│       ├── business_cost.py    # Framework de custo de negócio (FP vs FN)
│       ├── mlp_model.py        # Arquitetura MLP + treino com early stopping
│       ├── train.py            # Script de treino do modelo de produção
│       ├── schemas.py          # Schemas Pydantic (request/response da API)
│       ├── inference.py        # Carrega artefatos e executa predições
│       ├── model_registry.py   # Baixa artefatos do modelo via Cloud Storage (deploy)
│       ├── logging_config.py   # Logging estruturado (JSON), sem print()
│       ├── metrics.py          # Métricas operacionais (Prometheus) para /metrics
│       └── api.py              # API FastAPI (/, /health, /ready, /infer, /metadata, /metrics)
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Fixtures compartilhadas
│   ├── test_schema.py          # Validação de schema (pandera)
│   ├── test_smoke.py           # Smoke tests do pipeline/treino/inferência
│   ├── test_api.py             # Testes dos endpoints da API (FastAPI TestClient)
│   ├── test_model_registry.py  # Testes do model registry (modo local e bucket via mock)
│   └── test_streamlit_app.py   # Testes da lógica de integração do dashboard
├── Dockerfile                  # Imagem de produção da API
├── Dockerfile.streamlit        # Imagem de produção do dashboard Streamlit
├── cloudbuild.streamlit.yaml   # Config explícita do Cloud Build para o dashboard
├── requirements-api.txt        # Dependências mínimas de runtime da API
├── requirements-app.txt        # Dependências mínimas de runtime do dashboard
├── Makefile                    # Atalhos opcionais (Linux/macOS/WSL) — ver Setup do ambiente
├── .gitignore
├── .dockerignore
├── LICENSE                    # Licença MIT (código e artefatos originais do projeto)
├── NOTICE.md                  # Direitos autorais do material do desafio (FIAP)
├── pyproject.toml            # Single source of truth: deps, ruff, pytest
└── README.md
```

## Setup do ambiente

> ⚠️ **Use Python 3.10, 3.11 ou 3.12.** Versões mais novas (3.13/3.14) **não
> são suportadas** — o MLflow 3.x falha ao iniciar nessas versões
> (`ImportError: cannot import name 'Traversable' from 'importlib.abc'`),
> pois depende de uma API do `importlib.abc` removida no Python 3.13+. O
> `pyproject.toml` já restringe `requires-python` para evitar essa instalação,
> mas é importante **criar o ambiente virtual já com a versão certa**.

Pré-requisitos: **Python 3.10, 3.11 ou 3.12** (recomendado: 3.12), **Git**.

### 1. Clonar o repositório

```bash
git clone https://github.com/tiagotff/FIAP_MLE10_TC1.git
cd FIAP_MLE10_TC1
```

### 2. Criar o ambiente virtual com uma versão suportada

<details open>
<summary><b>🐧 Linux / 🍎 macOS</b></summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

</details>

<details open>
<summary><b>🪟 Windows (Git Bash / MINGW64)</b></summary>

Confira as versões de Python instaladas com `py -0`. Se não tiver 3.10–3.12,
baixe o instalador em https://www.python.org/downloads/release/python-3127/
(marque **"Add python.exe to PATH"** durante a instalação).

```bash
py -0                       # lista as versões instaladas
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

</details>

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Se aparecer um erro de política de execução de scripts, rode uma vez (como
administrador): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

</details>

<details>
<summary><b>🪟 Windows (cmd)</b></summary>

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

</details>

Em qualquer sistema, o prompt do terminal passa a exibir `(.venv)` quando o
ambiente está ativado corretamente.

### 3. Instalar o projeto e as dependências

```bash
pip install -e ".[dev]"
```

Isso instala todas as dependências declaradas no `pyproject.toml`: PyTorch,
Scikit-Learn, MLflow, FastAPI, Pydantic, Pandera, Prometheus client,
Google Cloud Storage, Streamlit, ferramentas de teste (`pytest`,
`pytest-cov`, `responses`) e lint (`ruff`).

> 💡 Como o projeto é instalado em modo editável (`pip install -e`), o
> pacote `churn_prediction` já fica importável globalmente no ambiente
> virtual — o `PYTHONPATH=src` usado nos comandos deste README é
> redundante após esse passo, mas é mantido explícito por clareza e como
> rede de segurança (funciona mesmo se o passo de instalação for pulado).

### Comandos do projeto (com ou sem `make`)

Todos os comandos abaixo funcionam em qualquer sistema, direto com
`python`/`pip`/`pytest` — **é essa a forma recomendada e usada ao longo
deste README.**

Para quem usa Linux, macOS, ou WSL no Windows, o projeto também inclui um
`Makefile` com atalhos equivalentes (`make install`, `make lint`, etc.) —
totalmente opcional. **No Windows (Git Bash/MINGW64/PowerShell/cmd), `make`
não vem instalado por padrão**; use os comandos diretos da tabela abaixo,
ou instale `make` via [Chocolatey](https://chocolatey.org/)
(`choco install make`) ou WSL caso prefira os atalhos.

| Tarefa | Comando direto (recomendado, qualquer SO) | Atalho `make` (Linux/macOS/WSL) |
|---|---|---|
| Instalar dependências | `pip install -e ".[dev]"` | `make install` |
| Lint | `ruff check .` | `make lint` |
| Formatar código | `ruff check --fix .` | `make format` |
| Rodar testes | ver [Testes automatizados](#testes-automatizados) | `make test` |
| Testes com cobertura | ver [Testes automatizados](#testes-automatizados) | `make test-cov` |
| Treinar o modelo | ver [API de inferência](#api-de-inferência) | `make train` |
| Rodar a API localmente | ver [API de inferência](#api-de-inferência) | `make run` |
| Rodar o dashboard localmente | ver [Dashboard Streamlit](#dashboard-streamlit) | `make run-app` |

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

### Notebook de modelagem com MLP (Etapa 2)

```bash
jupyter notebook notebooks/02_mlp_model_comparison.ipynb
```

Esse notebook:

1. Constrói e treina a MLP em PyTorch (arquitetura 64→32, ReLU, Dropout,
   `BCEWithLogitsLoss` com `pos_weight`, Adam, early stopping).
2. Adiciona um baseline de árvore (Random Forest), complementando o baseline
   linear da Etapa 1.
3. Compara 4 modelos (Dummy, Regressão Logística, Random Forest, MLP) no
   mesmo holdout de teste, com 6 métricas.
4. Aplica o framework de custo de negócio (FP vs. FN) e uma análise de
   sensibilidade ao custo de campanha.
5. Registra todos os experimentos e modelos (sklearn + PyTorch) no MLflow.

Para regenerar:

```bash
python scripts/generate_mlp_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_mlp_model_comparison.ipynb
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

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

E acesse `http://localhost:5001`.

**`bash: make: command not found` (Windows, Git Bash).**
O Windows não inclui `make` nativamente. Use o comando equivalente sem
`make` (documentado na tabela em [Setup do ambiente](#setup-do-ambiente)),
ou instale `make` via `choco install make` (Chocolatey) ou WSL.

**Dashboard Streamlit mostra "Não foi possível conectar à API".**
A API precisa estar rodando **em um terminal separado, ao mesmo tempo**
que o Streamlit — são dois processos independentes. Confirme com
`curl http://127.0.0.1:8000/ready` em um terceiro terminal: se der erro de
conexão, a API não está no ar. Veja
[API de inferência](#api-de-inferência) para subir a API antes do
dashboard, ou aponte `CHURN_API_URL` para a API já implantada em nuvem
(ver [Deploy em nuvem](#deploy-em-nuvem-bônus)) para testar sem precisar
rodar nada localmente.

## API de inferência

A partir da Etapa 3, o modelo escolhido para produção — a **MLP (PyTorch)**,
por ter o melhor recall e o melhor resultado no framework de custo de
negócio (ver [Resultados da Etapa 2](#resultados-da-etapa-2)) — é servido
via uma API FastAPI.

### 1. Treinar o modelo e gerar os artefatos

Antes de iniciar a API, é necessário treinar o modelo e salvar os artefatos
em `models/` (não versionados em git — apenas `model_metadata.json` é
versionado, por ser pequeno e legível).

<details open>
<summary><b>🐧🍎 Linux / macOS (bash/zsh) / 🪟 Windows (Git Bash)</b></summary>

```bash
PYTHONPATH=src python -m churn_prediction.train
```

</details>

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
$env:PYTHONPATH = "src"
python -m churn_prediction.train
```

</details>

<details>
<summary><b>🪟 Windows (cmd)</b></summary>

```cmd
set PYTHONPATH=src
python -m churn_prediction.train
```

</details>

Isso gera:
- `models/preprocessor.joblib` — pipeline de pré-processamento ajustado.
- `models/mlp_model.pt` — pesos treinados da MLP.
- `models/model_metadata.json` — métricas, parâmetros e versão do modelo.

### 2. Iniciar a API

```bash
# Linux/macOS ou Windows (Git Bash):
PYTHONPATH=src python -m uvicorn churn_prediction.api:app --reload

# Windows (PowerShell):
$env:PYTHONPATH = "src"; python -m uvicorn churn_prediction.api:app --reload

# Windows (cmd):
set PYTHONPATH=src && python -m uvicorn churn_prediction.api:app --reload
```

A API fica disponível em `http://127.0.0.1:8000`. Documentação interativa
(Swagger UI) em `http://127.0.0.1:8000/docs` — você pode testar todos os
endpoints abaixo direto no navegador, sem precisar de `curl`.

### Endpoints

A API segue a convenção adotada por plataformas de model serving em
produção (KServe, Seldon, BentoML, MLflow Serving): endpoints distintos
para liveness, readiness, inferência, metadados e métricas operacionais.

| Endpoint | Método | Propósito |
|---|---|---|
| `/` | GET | Informações básicas e links para os demais endpoints |
| `/health` | GET | **Liveness** — o processo está vivo? (não depende do modelo) |
| `/ready` | GET | **Readiness** — o modelo está carregado e pronto para inferência? |
| `/infer` | POST | Predição de churn para **um** cliente (nome canônico) |
| `/predict` | POST | Alias de `/infer`, mantido por compatibilidade |
| `/predict/batch` | POST | Predição para **até 500** clientes em uma chamada |
| `/metadata` | GET | Versão/métricas do modelo + JSON Schema de entrada/saída de `/infer` |
| `/metrics` | GET | Métricas operacionais da API no formato Prometheus |

**`GET /`** — informações básicas da API:

```bash
curl http://127.0.0.1:8000/
```

**`GET /health`** — liveness probe simples, sempre rápida:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

**`GET /ready`** — readiness probe: confirma que o modelo está carregado.
Diferente de `/health` — a API pode estar viva mas ainda não pronta (ex.:
durante o carregamento do modelo, ou se o treino nunca foi executado):

```bash
curl http://127.0.0.1:8000/ready
```

```json
{"status": "ready", "model_loaded": true}
```

**`GET /metadata`** — versão e métricas do modelo em produção, junto com o
JSON Schema esperado por `/infer` (útil para descobrir o contrato da API
programaticamente, sem ler a documentação):

```bash
curl http://127.0.0.1:8000/metadata
```

```json
{
  "model_info": {
    "model_version": "1.0.0",
    "trained_at": "2026-06-27T18:58:07.24Z",
    "test_roc_auc": 0.843,
    "test_recall": 0.778,
    "business_net_cost": -178606.4
  },
  "input_schema": { "...": "JSON Schema completo dos campos aceitos por /infer" },
  "output_schema": { "...": "JSON Schema da resposta de /infer" }
}
```

**`GET /metrics`** — métricas operacionais da API (contagem de
requisições, latência por rota, predições por nível de risco), no formato
texto do Prometheus — pronto para scraping por um servidor Prometheus ou
visualização em um dashboard Grafana:

```bash
curl http://127.0.0.1:8000/metrics
```

```
churn_api_requests_total{method="POST",path="/infer",status_code="200"} 12.0
churn_api_request_latency_seconds_sum{method="POST",path="/infer"} 0.58
churn_predictions_total{risk_level="high"} 7.0
churn_predictions_total{risk_level="low"} 5.0
```

**`POST /infer`** — recebe os dados de um cliente e retorna o risco de
churn (endpoint canônico de inferência; `POST /predict` é um alias
idêntico, mantido por compatibilidade):

<details open>
<summary><b>🐧🍎 Linux / macOS / 🪟 Windows (Git Bash)</b></summary>

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

</details>

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

O `curl` do PowerShell é um alias de `Invoke-WebRequest` com sintaxe
diferente — use `curl.exe` para o `curl` real, ou `Invoke-RestMethod`:

```powershell
$body = @{
  gender = "Female"; SeniorCitizen = 0; Partner = "Yes"; Dependents = "No"
  tenure = 1; PhoneService = "No"; MultipleLines = "No phone service"
  InternetService = "DSL"; OnlineSecurity = "No"; OnlineBackup = "Yes"
  DeviceProtection = "No"; TechSupport = "No"; StreamingTV = "No"
  StreamingMovies = "No"; Contract = "Month-to-month"; PaperlessBilling = "Yes"
  PaymentMethod = "Electronic check"; MonthlyCharges = 29.85; TotalCharges = 29.85
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/infer" -Method Post -ContentType "application/json" -Body $body
```

</details>

Resposta:

```json
{"churn_probability": 0.7829, "churn_prediction": true, "risk_level": "high", "model_version": "1.0.0"}
```

Entradas inválidas (ex.: uma categoria fora do domínio esperado, como
`"Contract": "Three years"`) são rejeitadas com **HTTP 422**, antes de
chegar à lógica de inferência — validação automática via Pydantic.

**`POST /predict/batch`** — mesma predição, para até 500 clientes em uma
única chamada (útil para pontuar uma carteira de clientes de uma vez):

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"customers": [ { "...": "mesmo formato do /infer, um objeto por cliente" } ]}'
```

```json
{"predictions": [{"churn_probability": 0.78, "...": "..."}], "count": 1}
```

### Outros detalhes da API

- Toda requisição é logada em formato estruturado (JSON), registrada nas
  métricas Prometheus (`/metrics`) e recebe os headers `X-Request-ID` e
  `X-Process-Time-Ms` (latência em milissegundos), via middleware.
- **CORS** habilitado — a API pode ser consumida diretamente do navegador
  por um frontend (ex.: dashboard de CRM).
- Qualquer erro inesperado (não relacionado à validação de entrada) é
  capturado por um handler genérico e retorna **HTTP 500** com uma mensagem
  padrão — detalhes internos (stack trace, exceção original) nunca são
  expostos na resposta, apenas registrados no log estruturado.

## Dashboard Streamlit

Um cliente visual da API (`app/streamlit_app.py`), pensado para um usuário
de negócio (time de Retenção/CRM) sem conhecimento técnico de APIs —
nenhuma lógica de ML roda neste app; ele apenas envia requisições à API e
exibe o resultado.

### Executar localmente

Com a API já rodando (Seção anterior) em outro terminal:

```bash
# Linux/macOS ou Windows (Git Bash):
CHURN_API_URL=http://127.0.0.1:8000 streamlit run app/streamlit_app.py

# Windows (PowerShell):
$env:CHURN_API_URL = "http://127.0.0.1:8000"
streamlit run app/streamlit_app.py

# Windows (cmd):
set CHURN_API_URL=http://127.0.0.1:8000 && streamlit run app/streamlit_app.py
```

O dashboard abre em `http://localhost:8501`. Se `CHURN_API_URL` não for
definida, o app tenta `http://127.0.0.1:8000` por padrão.

### O que o dashboard oferece

- **Sidebar**: status da API (`GET /ready`) e um resumo das métricas do
  modelo em produção (`GET /metadata`) — AUC-ROC, recall, custo de
  negócio líquido.
- **Aba "Cliente único"**: formulário completo (espelha
  `ChurnPredictionRequest`) para avaliar um cliente via `POST /infer`.
- **Aba "Carteira (CSV)"**: upload de um CSV com até 500 clientes,
  avaliados em uma única chamada a `POST /predict/batch`; mostra a
  distribuição de risco, uma tabela ordenada por probabilidade de churn,
  e permite baixar o resultado em CSV.

### Deploy em nuvem

O dashboard pode ser implantado no Cloud Run como um serviço independente
da API (`Dockerfile.streamlit`), apontando para a URL pública da API já
implantada:

```bash
./scripts/deploy_streamlit_to_cloud_run.sh SEU_PROJETO https://SUA-API.run.app
```

## Testes automatizados

```bash
# Linux/macOS ou Windows (Git Bash):
PYTHONPATH=src python -m pytest tests/ -v

# Windows (PowerShell):
$env:PYTHONPATH = "src"; python -m pytest tests/ -v

# Windows (cmd):
set PYTHONPATH=src && python -m pytest tests/ -v
```

Para incluir o relatório de cobertura, adicione
`--cov=src/churn_prediction --cov-report=term-missing` ao final de
qualquer um dos comandos acima.

A suíte cobre os 3 tipos de teste exigidos, com **38 testes** no total:

| Arquivo | Tipo | O que valida |
|---|---|---|
| `tests/test_schema.py` | Schema (pandera) | Domínio de valores categóricos, tipos, ausência de nulos no dataset bruto e pós-tratamento |
| `tests/test_smoke.py` | Smoke test | Pipeline de dados, pré-processamento, treino reduzido da MLP e inferência executam de ponta a ponta sem erros |
| `tests/test_api.py` | API (FastAPI TestClient) | `/`, `/health`, `/ready`, `/metadata`, `/metrics`, `/infer`, `/predict` (alias) e `/predict/batch` (válido, inválido, limite de 500), CORS, headers de latência, erro 500 genérico sem detalhes internos |
| `tests/test_model_registry.py` | Unitário (mocks) | Download dos artefatos do modelo via Cloud Storage (modo bucket) e comportamento no-op em desenvolvimento local (sem `MODEL_BUCKET`) |
| `tests/test_streamlit_app.py` | Unitário (mocks) | Resolução da URL da API, wrapper de chamadas HTTP do dashboard, e sincronização do schema do CSV de upload com `ChurnPredictionRequest` |

Os testes de `test_smoke.py` e `test_api.py` que dependem do modelo treinado
são automaticamente pulados (`pytest.skip`) se o treino (Seção
[API de inferência](#api-de-inferência)) ainda não tiver sido executado.

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

## Resultados da Etapa 2

Modelagem completa documentada em [`notebooks/02_mlp_model_comparison.ipynb`](notebooks/02_mlp_model_comparison.ipynb).

### Comparação de modelos (holdout de teste)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Acurácia |
|---|---|---|---|---|---|
| DummyClassifier | 0.516 | 0.272 | 0.290 | 0.291 | 0.622 |
| Regressão Logística | 0.841 | 0.633 | 0.614 | 0.783 | 0.738 |
| Random Forest | 0.839 | 0.650 | 0.625 | 0.714 | 0.772 |
| **MLP (PyTorch)** | 0.842 | 0.637 | 0.623 | **0.802** | 0.742 |

Os três modelos "reais" performam de forma muito próxima em AUC-ROC (~0.84)
— a MLP não traz ganho estatístico expressivo sobre os baselines neste
dataset tabular de porte moderado.

### Framework de custo de negócio (FP vs. FN)

| Modelo | Custo líquido (R$) |
|---|---|
| DummyClassifier | +153.073,60 (perda de valor) |
| Regressão Logística | −182.330,00 (ganho) |
| Random Forest | −130.438,00 (ganho, menor que os demais) |
| **MLP (PyTorch)** | **−192.946,00 (maior ganho)** |

A MLP se destaca no critério de negócio por ter o maior recall (80,2%),
reduzindo o número de falsos negativos — clientes que cancelam sem serem
identificados a tempo, o tipo de erro mais caro neste framework. Análise de
sensibilidade ao custo de campanha (R$ 10–200) confirma que esse ranking é
estável nessa faixa.

Detalhes completos do framework de custo: ver [`docs/ml_canvas.md`](docs/ml_canvas.md#11-aplicação-do-framework-de-custo-etapa-2).

## Resultados da Etapa 3

Refatoração completa do projeto em módulos testáveis, com pipeline
reprodutível, API de inferência e testes automatizados.

### O que foi entregue

- **Pipeline reprodutível** (`src/churn_prediction/pipeline.py`): além do
  `ColumnTransformer` (escala + one-hot), inclui um **transformador
  customizado** (`TenureBucketizer`), que deriva a feature `tenure_bucket`
  a partir do achado da EDA (Etapa 1) de que o churn cai de forma quase
  monotônica com o tempo de relacionamento.
- **Script de treino** (`src/churn_prediction/train.py`): consolida a
  decisão da Etapa 2 (MLP como modelo de produção) e gera os artefatos
  finais (`models/preprocessor.joblib`, `models/mlp_model.pt`,
  `models/model_metadata.json`).
- **API FastAPI** (`src/churn_prediction/api.py`), seguindo a convenção de
  plataformas de model serving em produção (KServe, Seldon, BentoML):
  - `GET /health`: liveness — confirma que o processo está vivo (rápido,
    não depende do modelo).
  - `GET /ready`: readiness — confirma que o modelo está carregado e a API
    está pronta para receber tráfego.
  - `POST /infer`: predição para um único cliente (nome canônico);
    `POST /predict` continua disponível como alias, por compatibilidade.
  - `POST /predict/batch`: predição em lote (até 500 clientes por chamada),
    processada de forma vetorizada (uma única passada pelo modelo).
  - `GET /metadata`: versão e métricas do modelo (AUC-ROC, recall, custo de
    negócio) + JSON Schema de entrada/saída de `/infer`.
  - `GET /metrics`: métricas operacionais (contagem de requisições,
    latência por rota, predições por nível de risco) no formato Prometheus.
  - Validação de entrada via Pydantic (schemas em `schemas.py`); middleware
    de latência (`X-Request-ID`, `X-Process-Time-Ms`); **CORS** habilitado
    para consumo direto a partir de um frontend/navegador; tratamento de
    erro dedicado quando o modelo não está disponível (HTTP 503) e handler
    genérico para qualquer exceção não tratada (HTTP 500, sem expor
    detalhes internos ao cliente — apenas no log estruturado).
- **Logging estruturado** (`src/churn_prediction/logging_config.py`): todo
  log da aplicação (treino e API) é emitido em JSON — nenhum módulo de
  produção usa `print()`.
- **Testes automatizados** (`tests/`), distribuídos entre os 3 tipos
  exigidos: schema (pandera), smoke test e testes de API — todos passando
  (26 testes nesta etapa; a suíte completa do projeto, somando as etapas
  seguintes, está em [Testes automatizados](#testes-automatizados)).
- **Makefile** com os targets `install`, `lint`, `test`, `test-cov`,
  `train`, `run` e `clean` — atalhos opcionais, com o comando equivalente
  documentado para quem não usa `make` (ex.: Windows sem WSL/Chocolatey).

### Validação de qualidade nesta etapa

```bash
ruff check .                              # All checks passed!
PYTHONPATH=src python -m pytest tests/    # 38 passed (suíte completa e atual do projeto)
```

## Resultados da Etapa 4

Documentação final do projeto: Model Card, arquitetura de deploy e plano
de monitoramento.

### O que foi entregue

- **[Model Card](docs/model_card.md)**: performance no holdout de teste
  (AUC-ROC 0.844, recall 0.810), análise de fairness por gênero,
  por `SeniorCitizen` e por tipo de contrato (com achado real de recall
  0.000 em contratos de 2 anos, devido a apenas 9 casos positivos no
  teste), limitações conhecidas, considerações éticas e cenários de falha
  mapeados contra o que já está implementado na API.
- **[Arquitetura de deploy](docs/deployment_architecture.md)**: decisão
  por uma arquitetura **híbrida** (batch diário + API real-time já
  construída na Etapa 3), com justificativa baseada nos dois modos de uso
  reais do stakeholder (CRM): priorização de carteira (batch) e consulta
  pontual durante atendimento (real-time).
- **[Plano de monitoramento](docs/monitoring_plan.md)**: métricas
  operacionais (já instrumentadas via `/metrics`) e de qualidade do
  modelo (via `/metadata`), thresholds de alerta, cadência de
  re-avaliação periódica (mensal, dado o atraso natural do rótulo de
  churn) e playbook de resposta para 4 cenários de incidente.
- **[Apresentação STAR](docs/Apresentacao_Tech_Challenge_1_MLE10_Churn.pptx)**:
  slides completos cobrindo os 4 elementos do método STAR (Situation,
  Task, Action, Result), incluindo o diagrama de arquitetura ponta a
  ponta e os resultados quantitativos do projeto.
- **[Vídeo de apresentação (5 min, método STAR)](docs/video_apresentacao_STAR_TC1_MLE.mp4)**:
  gravação cobrindo os 4 elementos do método STAR, com demonstração da
  API e do dashboard em produção.

## Deploy em nuvem (bônus)

A API foi implantada em produção real no **Google Cloud Platform**, via
**Cloud Run**, com o modelo carregado dinamicamente de um bucket do
**Cloud Storage** (model registry — ver
[arquitetura de deploy](docs/deployment_architecture.md)).

> ⚠️ A URL abaixo pode não estar mais ativa após o período de avaliação,
> já que o projeto GCP é de uso pessoal/educacional e pode ser desligado
> para evitar custos. O endpoint foi validado e está documentado com
> evidência de funcionamento (Seção abaixo).

**Endpoint público**: `https://churn-api-855490327597.us-central1.run.app`

**Documentação interativa (Swagger UI)**: `https://churn-api-855490327597.us-central1.run.app/docs`
— permite testar qualquer endpoint (`/infer`, `/predict/batch`, etc.) direto
no navegador, sem precisar de `curl` ou terminal.

### Componentes do deploy

- **Imagem Docker** ([`Dockerfile`](Dockerfile)): build mínimo, sem
  artefatos de modelo embutidos — apenas código-fonte (`src/`) e
  dependências de runtime ([`requirements-api.txt`](requirements-api.txt)),
  com PyTorch CPU-only instalado a partir do índice oficial (evita ~1.5GB
  de dependências CUDA desnecessárias).
- **Model registry** (`src/churn_prediction/model_registry.py`): a API
  baixa `mlp_model.pt` e `preprocessor.joblib` de um bucket GCS na
  inicialização, configurado via a variável de ambiente `MODEL_BUCKET` —
  um novo modelo treinado pode ser promovido a produção apenas
  atualizando o bucket (via [`scripts/upload_model_to_gcs.sh`](scripts/upload_model_to_gcs.sh)),
  sem rebuild ou redeploy da imagem.
- **Cloud Build**: builda e publica a imagem a partir do código-fonte.
- **Cloud Run**: serviço serverless, escala a zero quando sem tráfego —
  compatível com o free tier permanente do GCP para o volume esperado
  deste projeto.

### Evidência de funcionamento (todos os 8 endpoints testados em produção)

```bash
$ curl https://churn-api-855490327597.us-central1.run.app/ready
{"status":"ready","model_loaded":true}

$ curl -X POST https://churn-api-855490327597.us-central1.run.app/infer \
  -H "Content-Type: application/json" -d '{...}'
{"churn_probability":0.8032,"churn_prediction":true,"risk_level":"high","model_version":"1.0.0"}
```

`/metadata` e `/metrics` também foram validados — `/metrics` confirmou o
tráfego real gerado durante os testes
(`churn_api_requests_total`, latência por rota via histograma,
`churn_predictions_total{risk_level="high"}`).

### Reproduzindo o deploy

```bash
# 1. Criar projeto e habilitar APIs
gcloud projects create SEU_PROJETO --name="Churn Prediction"
gcloud config set project SEU_PROJETO
gcloud services enable run.googleapis.com cloudbuild.googleapis.com storage.googleapis.com

# 2. Criar bucket e subir o modelo treinado
#    (treine antes com: PYTHONPATH=src python -m churn_prediction.train)
gcloud storage buckets create gs://SEU_BUCKET --location=us-central1
./scripts/upload_model_to_gcs.sh SEU_BUCKET

# 3. Build e deploy da API
gcloud builds submit --tag gcr.io/SEU_PROJETO/churn-api
gcloud run deploy churn-api \
  --image gcr.io/SEU_PROJETO/churn-api \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_BUCKET=SEU_BUCKET \
  --memory 1Gi --cpu 1 --port 8080

# 4. (Opcional) Build e deploy do dashboard Streamlit, apontando para a API acima
./scripts/deploy_streamlit_to_cloud_run.sh SEU_PROJETO https://SUA-API.run.app
```

## Entrega final

Projeto concluído de ponta a ponta — todas as 4 etapas, incluindo o bônus
de deploy em nuvem:

- 🎥 **[Vídeo de apresentação (método STAR)](docs/video_apresentacao_STAR_TC1_MLE.mp4)**
- 📊 **[Slides de apoio](docs/Apresentacao_Tech_Challenge_1_MLE10_Churn.pptx)**
  (também disponíveis em [PDF](docs/Apresentacao_Tech_Challenge_1_MLE10_Churn.pdf))
- ☁️ **API em produção**: ver [Deploy em nuvem](#deploy-em-nuvem-bônus)

---

## Licença

O código-fonte e a documentação autoral deste repositório estão licenciados
sob a **Licença MIT** — ver [`LICENSE`](LICENSE).

O enunciado oficial do desafio, disponível em
[`docs/tech-challenge-fiap.pdf`](docs/tech-challenge-fiap.pdf), é de autoria
da **FIAP / POS TECH** e **não está** coberto pela licença MIT acima —
incluído apenas para fins de referência e contexto acadêmico. Detalhes em
[`NOTICE.md`](NOTICE.md).

---

## Equipe

Tech Challenge 1 — FIAP MLE10 - Machine Learning Engineering

By Tiago de Freitas Faustino
