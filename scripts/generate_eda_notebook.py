"""Gera o notebook 01_eda_baselines.ipynb célula por célula.

Não é parte do pacote de produção (src/); é uma ferramenta de build local
para montar o notebook de forma reprodutível a partir de código testado.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md("""\
# Tech Challenge 1 — Etapa 1: Entendimento e Preparação

**Previsão de Churn em Telecomunicações**

Este notebook cobre a Etapa 1 do projeto:
- Exploração de dados (EDA) completa: volume, qualidade, distribuição, data readiness.
- Definição de métricas técnicas e de negócio.
- Treinamento de baselines (`DummyClassifier` e Regressão Logística).
- Registro dos experimentos no MLflow.

> O ML Canvas do projeto está documentado separadamente em `docs/ml_canvas.md`.
""")

# ---------------------------------------------------------------------------
md("## 1. Setup e imports")

code("""\
import sys
sys.path.insert(0, "../src")

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RAW_DATASET_PATH,
    SEED,
    TARGET_COLUMN,
    set_global_seed,
)

set_global_seed(SEED)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
print(f"Seed fixada em: {SEED}")
""")

# ---------------------------------------------------------------------------
md("""\
## 2. Carga dos dados

Dataset: **Telco Customer Churn (IBM)** — disponível publicamente, contendo dados
demográficos, de serviços contratados e de cobrança de 7.043 clientes de uma
operadora de telecomunicações fictícia.
""")

code("""\
df = pd.read_csv(RAW_DATASET_PATH)
print(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")
df.head()
""")

# ---------------------------------------------------------------------------
md("""\
## 3. Data Readiness — Volume, Qualidade e Tipos

Critério mínimo do desafio: ≥ 5.000 registros e ≥ 10 features. O dataset
atende com folga (7.043 registros, 20 features + target).
""")

code("""\
df.info()
""")

code("""\
# Checagem de duplicados pelo identificador do cliente
print(f"customerID duplicados: {df['customerID'].duplicated().sum()}")

# Checagem de nulos explícitos (NaN)
print("\\nNulos (NaN) por coluna:")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else "Nenhum NaN encontrado.")
""")

md("""\
**Achado de qualidade de dados #1 — `TotalCharges`:** a coluna está tipada como
`object` em vez de numérica. Isso acontece porque alguns registros têm o valor
representado como string vazia (`" "`), e não como `NaN` — por isso o
`isnull()` acima não os captura.
""")

code("""\
# TotalCharges está como string. Forçando conversão numérica para revelar os problemas.
total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
mask_invalid = total_charges_numeric.isna()

print(f"Registros com TotalCharges não numérico: {mask_invalid.sum()}")
df.loc[mask_invalid, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]
""")

md("""\
**Explicação:** todos os 11 registros problemáticos têm `tenure == 0`, ou seja,
são clientes recém-cadastrados que ainda não completaram um ciclo de cobrança.
Faz sentido de negócio que `TotalCharges` seja vazio nesse caso — **não é erro
de coleta, é uma regra de negócio implícita.**

**Decisão de tratamento:** converter `TotalCharges` para numérico e preencher
esses 11 casos com `0` (consistente com `tenure=0`), em vez de descartar as
linhas (perderíamos sinal, ainda que pequeno).
""")

code("""\
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
print("Tipo de TotalCharges após correção:", df["TotalCharges"].dtype)
print("Nulos restantes:", df["TotalCharges"].isnull().sum())
""")

md("""\
**Achado de qualidade de dados #2 — `SeniorCitizen`:** está representado como
`int64` (0/1) enquanto as demais flags binárias (`Partner`, `Dependents`, etc.)
são strings `"Yes"/"No"`. Mantemos como está nesta etapa (o pipeline de
pré-processamento da Etapa 3 vai padronizar os tipos), mas registramos a
inconsistência.
""")

# ---------------------------------------------------------------------------
md("""\
## 4. Distribuição da variável-alvo (`Churn`)

Esse é o ponto mais importante da EDA para um problema de classificação: medir
o desbalanceamento de classes, que vai guiar a escolha de métrica técnica e a
estratégia de validação (estratificação obrigatória).
""")

code("""\
churn_counts = df[TARGET_COLUMN].value_counts()
churn_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100

print(churn_counts)
print()
print(churn_pct.round(2))

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2", legend=False, ax=ax)
ax.set_title("Distribuição da variável-alvo (Churn)")
ax.set_xlabel("Churn")
ax.set_ylabel("Quantidade de clientes")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.show()
""")

md("""\
**Achado central:** a base é **desbalanceada — aproximadamente 73,5% "No" vs.
26,5% "Yes"**. Isso tem implicações diretas:

- **Acurácia é uma métrica enganosa aqui**: um classificador "burro" que sempre
  prevê "No" já acerta ~73,5% das vezes sem aprender nada útil. Por isso o
  baseline `DummyClassifier` é incluído explicitamente — ele serve de "piso"
  de comparação.
- A validação cruzada **precisa ser estratificada** (`StratifiedKFold`), para
  que cada fold preserve a proporção de classes.
- Métricas mais informativas para este problema: **AUC-ROC, PR-AUC e F1**
  (a métrica técnica final é definida na Seção 5).
""")

# ---------------------------------------------------------------------------
md("## 5. Métrica técnica e métrica de negócio")

md("""\
### Métrica técnica
Dado o desbalanceamento de classes, adotamos como métricas técnicas principais:

- **AUC-ROC**: mede a capacidade do modelo de separar as classes
  independentemente do threshold escolhido. Boa para comparar modelos.
- **PR-AUC (Average Precision)**: mais sensível ao desempenho na classe
  minoritária (churn) do que a AUC-ROC, especialmente relevante em bases
  desbalanceadas.
- **F1-score**: equilíbrio entre precisão e recall em um threshold fixo,
  relevante para a decisão operacional final (quem entra na campanha de
  retenção).

### Métrica de negócio
A diretoria quer reduzir o **custo de churn evitado**. Propomos um framework
de custo simples, parametrizável, baseado na matriz de confusão:

- **Falso Negativo (FN)** — cliente que vai cancelar e o modelo não identificou:
  custo = perda da receita do cliente (estimamos como `12 × MonthlyCharges`,
  ou seja, um ano de receita perdida).
- **Falso Positivo (FP)** — cliente que não ia cancelar, mas o modelo sinalizou
  risco e a empresa investiu em retenção (desconto, ligação, oferta): custo
  estimado como um valor fixo de campanha (ex.: R$ 50,00 por contato).
- **Verdadeiro Positivo (TP)** — cliente que ia cancelar e foi retido a tempo:
  ganho = receita preservada menos custo da campanha.

Esse framework será aplicado na Etapa 2, quando comparamos modelos sob a
ótica de custo, não apenas de métricas estatísticas.
""")

# ---------------------------------------------------------------------------
md("## 6. Distribuições das variáveis numéricas")

code("""\
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[numeric_cols].describe().T
""")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(data=df, x=col, hue=TARGET_COLUMN, kde=True, ax=ax, palette="Set2", element="step")
    ax.set_title(f"Distribuição de {col}")
plt.tight_layout()
plt.show()
""")

md("""\
**Achados:**
- `tenure` (meses de contrato) tem uma concentração visível de clientes com
  pouco tempo de casa, e o churn é claramente mais frequente nessa faixa
  (confirmado na análise de churn por faixa de tenure, abaixo).
- `MonthlyCharges` mostra uma cauda de clientes com cobranças mais altas tendo
  maior proporção de churn.
- `TotalCharges` é fortemente correlacionado com `tenure` (cliente mais antigo
  acumula mais cobrança total) — candidato a feature redundante/colinear.
""")

# ---------------------------------------------------------------------------
md("## 7. Churn por variáveis categóricas-chave")

code("""\
def churn_rate_by(col):
    return df.groupby(col)[TARGET_COLUMN].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, col in zip(axes.flat, ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]):
    rates = churn_rate_by(col)
    sns.barplot(x=rates.values, y=rates.index, ax=ax, palette="rocket", hue=rates.index, legend=False)
    ax.set_xlabel("Taxa de churn")
    ax.set_title(f"Churn rate por {col}")
    for i, v in enumerate(rates.values):
        ax.text(v, i, f" {v:.1%}", va="center")

plt.tight_layout()
plt.show()
""")

md("""\
**Achados de negócio mais relevantes:**

- **Tipo de contrato é o fator mais discriminante**: clientes `Month-to-month`
  têm taxa de churn de **~42,7%**, contra **~11,3%** em contratos de 1 ano e
  apenas **~2,8%** em contratos de 2 anos. Sugere fortemente que o **tipo de
  contrato deve ser feature central** no modelo.
- **Fibra óptica tem o dobro do churn da DSL** (~41,9% vs. ~19,0%), e clientes
  sem internet têm a menor taxa (~7,4%) — possivelmente porque o serviço de
  fibra é mais caro e/ou tem mais problemas de qualidade percebida.
- **Electronic check é o método de pagamento com churn muito mais alto**
  (~45,3%) comparado a métodos automáticos (cartão de crédito ~15,2%,
  transferência bancária ~16,7%) — pode indicar um perfil de cliente menos
  engajado ou processos de pagamento mais sujeitos a fricção/inadimplência.
""")

code("""\
df["tenure_bucket"] = pd.cut(
    df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
)
tenure_churn = df.groupby("tenure_bucket", observed=True)[TARGET_COLUMN].apply(lambda s: (s == "Yes").mean())

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=tenure_churn.index, y=tenure_churn.values, hue=tenure_churn.index, palette="rocket", legend=False, ax=ax)
ax.set_ylabel("Taxa de churn")
ax.set_xlabel("Faixa de tenure (meses)")
ax.set_title("Churn rate por tempo de relacionamento (tenure)")
for i, v in enumerate(tenure_churn.values):
    ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

df.drop(columns=["tenure_bucket"], inplace=True)
""")

md("""\
**Achado:** a taxa de churn cai de forma quase monotônica com o tempo de
relacionamento — de **~47,4%** nos primeiros 12 meses para **~9,5%** após 49
meses. Isso é consistente com o padrão clássico de "churn de boas-vindas":
clientes recém-adquiridos são o segmento de maior risco e o foco natural de
qualquer estratégia de retenção precoce.
""")

# ---------------------------------------------------------------------------
md("## 8. Correlação entre variáveis numéricas")

code("""\
corr_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
corr_df["Churn_bin"] = (df[TARGET_COLUMN] == "Yes").astype(int)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlação (Pearson)")
plt.tight_layout()
plt.show()
""")

md("""\
**Achado:** `tenure` e `TotalCharges` são fortemente correlacionados (esperado,
já que `TotalCharges` é aproximadamente o produto de `tenure` por
`MonthlyCharges`). Isso é um ponto de atenção para a Etapa 2 — colinearidade
pode afetar a interpretabilidade de modelos lineares, embora tenha menor
impacto na rede neural (MLP). `tenure` tem a maior correlação (negativa) com
o churn entre as variáveis numéricas.
""")

# ---------------------------------------------------------------------------
md("""\
## 9. Preparação para modelagem: split treino/teste

Reservamos um holdout de teste (20%) **estratificado** pela variável-alvo, que
não será tocado até a avaliação final de modelos na Etapa 2.
""")

code("""\
X = df.drop(columns=["customerID", TARGET_COLUMN])
y = (df[TARGET_COLUMN] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
print(f"Taxa de churn no treino: {y_train.mean():.3f} | Taxa de churn no teste: {y_test.mean():.3f}")
""")

# ---------------------------------------------------------------------------
md("""\
## 10. Pipeline de pré-processamento (Scikit-Learn)

Pipeline simples reutilizável: imputação + escala para numéricas, one-hot
para categóricas. Este mesmo `ColumnTransformer` será reaproveitado como base
do pipeline refatorado na Etapa 3.
""")

code("""\
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_features),
    ]
)
preprocessor
""")

# ---------------------------------------------------------------------------
md("""\
## 11. Baselines: DummyClassifier e Regressão Logística

Treinamos dois baselines com **validação cruzada estratificada (k=5)**:

1. **DummyClassifier** (estratégia `stratified`): o "piso" — gera previsões
   respeitando a proporção das classes, sem nenhum aprendizado real. Qualquer
   modelo que não supere isso não tem valor preditivo.
2. **Regressão Logística**: baseline linear interpretável, comum em modelos
   de churn na indústria.
""")

code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def evaluate_cv(pipeline, X, y, cv):
    \"\"\"Avalia um pipeline via CV estratificada, retornando métricas médias.\"\"\"
    metrics = {"roc_auc": [], "pr_auc": [], "f1": [], "precision": [], "recall": [], "accuracy": []}

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
        metrics["pr_auc"].append(average_precision_score(y_val, y_proba))
        metrics["f1"].append(f1_score(y_val, y_pred))
        metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_pred))
        metrics["accuracy"].append(accuracy_score(y_val, y_pred))

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
""")

code("""\
dummy_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="stratified", random_state=SEED)),
])

dummy_metrics = evaluate_cv(dummy_pipeline, X_train, y_train, cv)

print("DummyClassifier — métricas médias (CV 5-fold estratificada):")
for metric, (mean, std) in dummy_metrics.items():
    print(f"  {metric:10s}: {mean:.4f} +/- {std:.4f}")
""")

code("""\
logreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
])

logreg_metrics = evaluate_cv(logreg_pipeline, X_train, y_train, cv)

print("Regressão Logística — métricas médias (CV 5-fold estratificada):")
for metric, (mean, std) in logreg_metrics.items():
    print(f"  {metric:10s}: {mean:.4f} +/- {std:.4f}")
""")

md("""\
**Leitura esperada dos resultados:** a Regressão Logística deve superar
claramente o `DummyClassifier` em AUC-ROC e PR-AUC, validando que há sinal
preditivo real nos dados — esse é o "gate" mínimo de qualidade antes de
avançarmos para a rede neural na Etapa 2.
""")

# ---------------------------------------------------------------------------
md("""\
## 12. Registro dos experimentos no MLflow

Registramos parâmetros, métricas e a versão do dataset para os dois baselines,
estabelecendo o histórico de experimentos que será expandido na Etapa 2 com a
MLP em PyTorch.
""")

code("""\
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

dataset_version = "telco-customer-churn-ibm-v1"


def log_baseline_run(run_name, pipeline, metrics_dict, extra_params=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("seed", SEED)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_features_raw", X_train.shape[1])
        mlflow.log_param("cv_strategy", "StratifiedKFold(n_splits=5)")
        if extra_params:
            mlflow.log_params(extra_params)

        for metric, (mean, std) in metrics_dict.items():
            mlflow.log_metric(f"{metric}_mean", mean)
            mlflow.log_metric(f"{metric}_std", std)

        # Treina no full train set para registrar o modelo como artefato
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, name="model")

        print(f"Run '{run_name}' registrada no MLflow.")


log_baseline_run(
    "baseline_dummy_classifier",
    dummy_pipeline,
    dummy_metrics,
    extra_params={"strategy": "stratified"},
)

log_baseline_run(
    "baseline_logistic_regression",
    logreg_pipeline,
    logreg_metrics,
    extra_params={"class_weight": "balanced", "max_iter": 1000},
)
""")

code("""\
print(f"Experimentos registrados em: {MLFLOW_TRACKING_URI}")
print("Para visualizar: execute `mlflow ui --backend-store-uri", MLFLOW_TRACKING_URI, "` na raiz do projeto.")
""")

# ---------------------------------------------------------------------------
md("""\
## 13. Resumo da Etapa 1 e próximos passos

**Resumo dos achados:**

1. Dataset com 7.043 registros e 20 features, sem nulos explícitos, mas com
   11 registros de `TotalCharges` vazio (clientes com `tenure=0`) — tratado
   via imputação para `0`.
2. Target desbalanceado (~73,5% / ~26,5%) → exige CV estratificada e métricas
   robustas a desbalanceamento (AUC-ROC, PR-AUC, F1), não apenas acurácia.
3. Variáveis com maior poder discriminante identificadas: `Contract`,
   `InternetService`, `tenure`, `PaymentMethod`.
4. Baselines (`DummyClassifier` e Regressão Logística) treinados, avaliados
   via CV estratificada e registrados no MLflow.

**Próximos passos (Etapa 2):** construir e treinar a MLP em PyTorch, comparar
com os baselines desta etapa usando ≥ 4 métricas, e analisar o trade-off de
custo entre falsos positivos e falsos negativos.
""")

nb["cells"] = cells

with open("/home/claude/FIAP_MLE10_TC1/notebooks/01_eda_baselines.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook gerado com sucesso.")
