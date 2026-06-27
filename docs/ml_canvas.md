# ML Canvas — Previsão de Churn de Clientes (Telecomunicações)

> Baseado no [Machine Learning Canvas](https://www.louisdorard.com/machine-learning-canvas), adaptado ao
> contexto do Tech Challenge 1 (FIAP — Pós Tech).

## 1. Proposta de Valor

Fornecer à diretoria de uma operadora de telecomunicações um modelo capaz de
**identificar, com antecedência, quais clientes têm maior risco de cancelar o
serviço (churn)**, permitindo que campanhas de retenção sejam direcionadas a
quem realmente importa — em vez de campanhas genéricas e caras para toda a
base de clientes.

## 2. Stakeholders

| Stakeholder | Interesse |
|---|---|
| **Diretoria / C-level** | Reduzir a taxa de churn e o custo associado à perda de receita recorrente. |
| **Time de Retenção/CRM** | Receber uma lista priorizada de clientes em risco para agir (ofertas, contato proativo). |
| **Time de Engenharia de Dados/ML** | Manter um pipeline confiável, versionado e monitorável. |
| **Clientes finais** | Beneficiados indiretamente por ofertas/atendimento personalizado antes de decidirem cancelar. |
| **Time de Compliance/Privacidade** | Garantir uso adequado dos dados pessoais e ausência de discriminação indevida (ver Model Card, Etapa 4). |

## 3. Usuários finais do modelo

Operadores do time de CRM/Retenção, via um relatório periódico (batch) ou uma
chamada de API (`/infer`) integrada ao CRM, que consultam o **score de risco
de churn** de cada cliente.

## 4. Objetivo de negócio (Business Goal)

Reduzir o **custo líquido do churn evitável** — ou seja, maximizar o valor de
clientes retidos com sucesso, descontado o custo das campanhas de retenção
aplicadas (inclusive aos falsos positivos).

### Métrica de negócio proposta
**Custo total estimado** = `(FN × custo_perda_cliente) + (FP × custo_campanha) − (TP × valor_retencao)`

Onde, nesta fase inicial:
- `custo_perda_cliente ≈ 12 × MonthlyCharges` (proxy para receita anual perdida).
- `custo_campanha` = custo fixo estimado de contato/oferta de retenção (parametrizável).
- `valor_retencao` = receita futura preservada por um TP, líquida do custo da campanha.

Esse framework de custo será aplicado formalmente na Etapa 2, ao comparar
modelos.

## 5. SLOs (Service Level Objectives)

| SLO | Meta inicial |
|---|---|
| **Latência de inferência (API)** | p95 < 300 ms por requisição (`/infer`) |
| **Disponibilidade da API** | ≥ 99% (ambiente de demonstração) |
| **Atualização do modelo** | Re-treinamento sob demanda / mensal (a definir na Etapa 4 — plano de monitoramento) |
| **Métrica técnica mínima para produção** | AUC-ROC ≥ 0.80 no holdout de teste |

## 6. Métricas técnicas (Offline)

Definidas a partir da EDA (Etapa 1), dado o desbalanceamento de classes
(~73,5% "No churn" vs. ~26,5% "Churn"):

- **AUC-ROC** — métrica primária de ranqueamento/discriminação.
- **PR-AUC (Average Precision)** — mais sensível à classe minoritária (churn).
- **F1-score** — equilíbrio precisão/recall no threshold de operação.
- *(Etapa 2 adiciona uma 4ª métrica explícita na comparação MLP vs. baselines, conforme exigido pelo desafio.)*

### Resultados de referência (baselines, Etapa 1 — CV 5-fold estratificada)

| Modelo | AUC-ROC | PR-AUC | F1 | Acurácia |
|---|---|---|---|---|
| DummyClassifier (estratificado) | 0.507 | 0.269 | 0.276 | 0.614 |
| Regressão Logística (`class_weight=balanced`) | 0.846 | 0.660 | 0.629 | 0.749 |

> A Regressão Logística supera com folga o piso aleatório do `DummyClassifier`,
> confirmando sinal preditivo real nos dados — pré-requisito para avançar com
> a rede neural na Etapa 2.

### Resultados finais (Etapa 2 — holdout de teste, 4 modelos)

| Modelo | AUC-ROC | PR-AUC | F1 | Recall | Acurácia |
|---|---|---|---|---|---|
| DummyClassifier | 0.516 | 0.272 | 0.290 | 0.291 | 0.622 |
| Regressão Logística | 0.841 | 0.633 | 0.614 | 0.783 | 0.738 |
| Random Forest | 0.839 | 0.650 | 0.625 | 0.714 | 0.772 |
| **MLP (PyTorch)** | 0.842 | 0.637 | 0.623 | **0.802** | 0.742 |

**Achado central da Etapa 2**: os três modelos "reais" performam de forma
muito próxima em AUC-ROC (~0.84) — a complexidade adicional da MLP não se
traduz em ganho estatístico expressivo neste dataset tabular de porte
moderado (~7k registros). Porém, ao aplicar o framework de custo de negócio
(Seção 6), a MLP se destaca por ter o **maior recall (80,2%)**, o que reduz
o número de falsos negativos (clientes que cancelam sem serem identificados)
— e isso pesa mais no custo de negócio do que pequenas variações de AUC.

## 7. Dados disponíveis

- **Fonte**: [Telco Customer Churn Dataset (IBM)](https://github.com/IBM/telco-customer-churn-on-icp4d) — dataset público.
- **Volume**: 7.043 registros, 20 features + variável-alvo (`Churn`).
- **Tipos de feature**: demográficas (gênero, idoso, dependentes), de
  serviços contratados (internet, telefonia, streaming, suporte técnico) e
  financeiras (tipo de contrato, forma de pagamento, cobrança mensal/total).
- **Qualidade**: sem `NaN` explícitos, mas 11 registros com `TotalCharges`
  vazio (clientes com `tenure=0`) — tratado via imputação.

## 8. Features candidatas (achados da EDA)

Variáveis com maior poder discriminante observado, por ordem de relevância
aparente:

1. `Contract` (mês-a-mês: ~42,7% de churn vs. 2 anos: ~2,8%)
2. `tenure` (0–12 meses: ~47,4% de churn vs. 49–72 meses: ~9,5%)
3. `InternetService` (fibra óptica: ~41,9% vs. sem internet: ~7,4%)
4. `PaymentMethod` (cheque eletrônico: ~45,3% vs. métodos automáticos: ~15–17%)
5. `MonthlyCharges`, `TotalCharges` — correlacionadas entre si (colinearidade a
   tratar na modelagem).

## 9. Riscos e vieses conhecidos (preliminar)

- **Desbalanceamento de classes** pode levar o modelo a favorecer a classe
  majoritária se não tratado adequadamente (mitigação: CV estratificada,
  `class_weight`/ponderação de loss, métricas robustas).
- **Dataset estático e de um único período/operadora fictícia** — pode não
  generalizar para sazonalidades reais, promoções específicas ou mudanças
  de mercado.
- **Risco de viés indireto**: variáveis demográficas (ex. `SeniorCitizen`,
  `gender`) podem correlacionar-se com churn por motivos socioeconômicos não
  causais — a ser detalhado no Model Card (Etapa 4) antes de qualquer uso real
  em decisões que afetem clientes.

## 10. Linha-base de decisão (build vs. baseline)

O modelo só avança para produção se superar consistentemente os baselines
acima (Regressão Logística) nas métricas técnicas E no framework de custo de
negócio — caso contrário, o ganho de complexidade da rede neural não se
justifica.

## 11. Aplicação do framework de custo (Etapa 2)

O framework de custo definido na Seção 4 foi aplicado aos 4 modelos no mesmo
holdout de teste (custo de campanha assumido: R$ 50,00; horizonte de receita:
12 meses):

| Modelo | Custo líquido (R$) | Leitura |
|---|---|---|
| DummyClassifier | +153.073,60 | Custo líquido positivo = perda de valor (não direciona campanhas de forma útil) |
| Regressão Logística | −182.330,00 | Ganho líquido |
| Random Forest | −130.438,00 | Ganho líquido, mas menor (recall mais baixo → mais FN caros) |
| **MLP (PyTorch)** | **−192.946,00** | **Maior ganho líquido** entre os modelos avaliados |

Uma análise de sensibilidade (variando o custo de campanha de R$ 10 a R$ 200)
mostrou que o ranking de negócio (MLP > Regressão Logística > Random Forest)
se mantém estável nessa faixa — um indício de robustez da conclusão, embora o
framework ainda dependa de calibração com custos reais de campanha antes de
uma decisão definitiva de produção.

**Conclusão da Etapa 2**: apesar de AUC-ROC quase empatada entre os 3
modelos "reais", a MLP é a candidata mais forte a seguir para a Etapa 3,
dado o maior recall e o melhor resultado no framework de custo de negócio —
sem perder de vista que a Regressão Logística continua sendo uma alternativa
competitiva e mais simples/interpretável, relevante para a decisão final de
arquitetura de deploy na Etapa 4.

## 12. Modelo de produção e API (Etapa 3)

A **MLP (PyTorch)** foi confirmada como modelo de produção, consolidando a
conclusão da Etapa 2. O modelo é servido via uma API FastAPI síncrona
(`/infer`), adequada ao SLO de latência definido na Seção 5
(p95 < 300ms) — a inferência de um único cliente em CPU leva poucos
milissegundos, dado o tamanho moderado da rede (64→32 neurônios).

### Decisões de engenharia desta etapa

- **Pipeline com transformador customizado**: a feature `tenure_bucket`
  (faixas de tempo de relacionamento) foi formalizada em um transformador
  sklearn (`TenureBucketizer`), reaproveitável entre treino e inferência —
  elimina o risco de divergência entre a lógica usada no notebook de
  exploração e a lógica usada em produção.
- **Validação de entrada estrita**: cada campo categórico do payload da API
  é validado contra o domínio exato observado na EDA (Etapa 1), via
  `Literal` do Pydantic — uma categoria nunca vista pelo modelo (ex.: um
  novo método de pagamento) é rejeitada com HTTP 422 antes de chegar à
  inferência, evitando previsões silenciosamente erradas.
- **Modelo não versionado em git, mas reprodutível**: os pesos do modelo
  (`mlp_model.pt`) e o pipeline ajustado (`preprocessor.joblib`) não são
  versionados (são artefatos binários grandes e regeráveis), mas
  `model_metadata.json` é versionado — documentando métricas, parâmetros e
  versão do modelo que gerou aqueles artefatos, mesmo sem os pesos em si.
- **Faixas de risco operacional**: a probabilidade contínua é também
  traduzida em `low`/`medium`/`high` (thresholds em 0.3 e 0.6), facilitando
  a priorização de campanhas pelo time de Retenção/CRM (stakeholder
  definido na Seção 2) sem que precisem interpretar uma probabilidade bruta.
- **Predição em lote (`/predict/batch`)**: além da predição unitária
  (`/infer`), o time de Retenção/CRM pode pontuar até 500 clientes em uma
  única chamada, processada de forma vetorizada (uma única passada pelo
  modelo) — relevante para rotinas batch periódicas de priorização de
  carteira.
- **Liveness vs. readiness (`/health` vs. `/ready`)**: separados conforme
  a convenção de plataformas de model serving — `/health` confirma apenas
  que o processo está vivo (sinal para um orquestrador não reiniciar o
  container), enquanto `/ready` confirma que o modelo está carregado e a
  API está de fato apta a atender chamadas de inferência (sinal para um
  load balancer rotear tráfego). Misturar as duas coisas em um único
  endpoint causaria reinícios desnecessários do container sempre que o
  modelo demorasse para carregar ou estivesse temporariamente indisponível.
- **Observabilidade em duas camadas (`/metadata` vs. `/metrics`)**:
  `/metadata` expõe a qualidade do **modelo** (AUC-ROC, recall, custo de
  negócio — a mesma informação que estava, em versões anteriores deste
  projeto, dentro do `/health`), enquanto `/metrics` expõe a saúde
  operacional da **API** (volume de requisições, latência por rota,
  distribuição de predições por nível de risco), no formato Prometheus —
  pronto para scraping por um servidor Prometheus/Grafana em um ambiente
  real de produção.
- **Falha segura (fail-safe) em erros inesperados**: qualquer exceção não
  tratada pela API retorna HTTP 500 com uma mensagem genérica — detalhes
  internos (stack trace, tipo da exceção) nunca são expostos ao cliente,
  apenas registrados no log estruturado para investigação.
