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
chamada de API (`/predict`) integrada ao CRM, que consultam o **score de risco
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
| **Latência de inferência (API)** | p95 < 300 ms por requisição (`/predict`) |
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
