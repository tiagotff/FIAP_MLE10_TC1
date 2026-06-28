# Arquitetura de Deploy — Batch vs. Real-time

## Decisão: arquitetura híbrida

Após avaliar os padrões de uso esperados pelo time de Retenção/CRM
(stakeholder definido no [ML Canvas](ml_canvas.md#2-stakeholders)), a
arquitetura de deploy escolhida é **híbrida**: combina um pipeline batch
periódico com a API de inferência em tempo real já construída na Etapa 3.

## Justificativa

Nem o batch puro nem o real-time puro cobrem adequadamente os dois modos
de uso reais deste problema de negócio:

| Caso de uso | Cobertura por Batch puro | Cobertura por Real-time puro |
|---|---|---|
| Lista priorizada diária de clientes em risco, para o time de CRM agir proativamente | ✅ Ideal | ⚠️ Desnecessariamente caro (pontuar toda a base via chamadas individuais) |
| Atendente consultando o risco de um cliente específico durante uma ligação | ❌ Não atende (dados podem estar desatualizados há horas/dias) | ✅ Ideal |
| Pontuação de um cliente recém-cadastrado, antes do próximo ciclo batch | ❌ Não atende | ✅ Ideal |
| Auditoria/reprocessamento de toda a base após uma mudança de modelo | ✅ Ideal (processa volume grande de forma eficiente) | ⚠️ Tecnicamente possível via `/predict/batch`, mas não é o caso de uso primário |

A arquitetura de código já construída nas Etapas 2 e 3 **já reflete essa
necessidade híbrida de forma natural**, não é uma decisão nova e
desconectada do que foi implementado:

- `POST /infer` — pontuação individual, latência baixa (p95 < 300ms, SLO
  definido no ML Canvas), pensada para consulta on-demand.
- `POST /predict/batch` — pontuação vetorizada de até 500 clientes por
  chamada, pensada para rotinas de carteira completa.
- `GET /health` / `GET /ready` — seguem a convenção de liveness/readiness
  usada por orquestradores de produção, compatível com ambos os modos de
  deploy (a mesma API serve tanto chamadas batch programáticas quanto
  chamadas real-time pontuais).

## Componentes da arquitetura

```
                    ┌─────────────────────────┐
                    │   models/ (artefatos)    │
                    │  preprocessor.joblib     │
                    │  mlp_model.pt            │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │                                     │
   ┌──────────▼───────────┐              ┌──────────▼───────────┐
   │   Pipeline Batch      │              │   API Real-time       │
   │   (diário, agendado)  │              │   (FastAPI, sempre-on)│
   │                        │              │                        │
   │ 1. Carrega carteira    │              │ GET  /health           │
   │    ativa de clientes   │              │ GET  /ready             │
   │ 2. POST /predict/batch │              │ POST /infer             │
   │    (chunks de 500)     │              │ POST /predict (alias)   │
   │ 3. Grava scores em     │              │ POST /predict/batch     │
   │    tabela/dashboard    │              │ GET  /metadata           │
   │ 4. Alerta o time de    │              │ GET  /metrics             │
   │    CRM (top N em risco)│              │                            │
   └────────────────────────┘              └────────────────────────────┘
              │                                          │
              ▼                                          ▼
   ┌────────────────────────┐              ┌────────────────────────────┐
   │  Dashboard / planilha   │              │  CRM / sistema de           │
   │  de priorização          │              │  atendimento consultando    │
   │  (time de Retenção)      │              │  score pontual               │
   └────────────────────────┘              └────────────────────────────┘
```

### 1. Pipeline Batch

- **Frequência sugerida**: diária (madrugada, fora do horário de pico de
  atendimento), revisável conforme a cadência real de atualização
  cadastral da operadora.
- **Mecanismo de execução**: um job agendado (cron, Airflow, ou um
  scheduler nativo da nuvem escolhida) que:
  1. Extrai a carteira de clientes ativos (de um data warehouse ou export
     do CRM).
  2. Divide em lotes de até 500 registros (limite atual do schema
     `BatchChurnPredictionRequest`).
  3. Chama `POST /predict/batch` para cada lote.
  4. Persiste os scores resultantes em uma tabela/planilha consultável
     pelo time de Retenção, ordenada por `risk_level`/`churn_probability`.
  5. Opcionalmente, dispara um alerta (e-mail/Slack) destacando os
     clientes que migraram para `risk_level=high` desde a última rodada.
- **Por que não chamar o modelo direto em batch (sem passar pela API)**:
  manter o batch como cliente da mesma API real-time garante que **um
  único caminho de código** (pipeline de pré-processamento + modelo)
  processe todas as predições, eliminando o risco de divergência entre a
  lógica batch e a lógica real-time — um problema comum quando essas duas
  vias são implementadas separadamente.

### 2. API Real-time

- Já implementada e testada na Etapa 3 (ver [README](../README.md#api-de-inferência)).
- Sempre ativa (`uvicorn`/processo gerenciado por um orquestrador), pronta
  para consultas pontuais de baixa latência.
- Mesmos artefatos de modelo (`models/preprocessor.joblib`,
  `models/mlp_model.pt`) usados pelo pipeline batch — **fonte única de
  verdade do modelo em produção**.

## Trade-offs assumidos

- **Consistência temporal**: o score visto no dashboard batch pode estar
  até 24h desatualizado em relação ao score real-time mais recente
  (aceitável para o caso de uso de priorização de campanha, mas é uma
  limitação explícita).
- **Complexidade operacional**: manter dois modos de acesso (job agendado
  + API sempre ativa) é mais complexo que apenas um dos dois — julgado
  como custo aceitável dado que ambos os casos de uso (Seção "Justificativa")
  são reais e relevantes para o stakeholder.
- **Sem feature store dedicada**: nesta fase, tanto o batch quanto o
  real-time recebem os dados já formatados pelo chamador (export do CRM,
  ou payload da API) — não há uma camada de feature engineering
  centralizada e versionada (ex.: Feast). Isso é aceitável no volume atual
  do projeto, mas seria o próximo investimento de infraestrutura caso o
  número de features ou a frequência de atualização crescesse.

## Evolução futura (fora do escopo desta fase)

- Migrar o agendamento batch para um orquestrador de workflows (Airflow,
  Dagster) com retries e observabilidade nativos, em vez de um cron
  simples.
- Adicionar uma fila de mensagens (ex.: SQS, Pub/Sub) entre eventos de
  negócio (ex.: "cliente alterou plano") e uma chamada automática a
  `/infer`, aproximando-se de um modelo "event-driven" sem exigir
  polling constante.
- Avaliar a viabilidade de detecção de *data drift* automatizada
  (ver [plano de monitoramento](monitoring_plan.md)) antes de expandir o
  uso do modelo para decisões de maior impacto.

## Validação em produção real (GCP)

O componente de **API Real-time** desta arquitetura foi implantado e
validado em um ambiente real de nuvem (Google Cloud Run), confirmando que
o desenho proposto funciona na prática, não apenas no papel:

- A API foi containerizada (`Dockerfile`) e implantada via Cloud Run,
  servindo publicamente em `https://churn-api-855490327597.us-central1.run.app`.
- Os artefatos do modelo foram desacoplados da imagem de deploy: o
  `model_registry.py` baixa `mlp_model.pt` e `preprocessor.joblib` de um
  bucket do Cloud Storage na inicialização (configurado via
  `MODEL_BUCKET`), validando exatamente a separação entre "ciclo de vida
  do treino" e "ciclo de vida do deploy de código" descrita na Seção
  "Componentes da arquitetura" acima.
- Todos os 8 endpoints (`/`, `/health`, `/ready`, `/infer`, `/predict`,
  `/predict/batch`, `/metadata`, `/metrics`) responderam corretamente em
  produção, incluindo a confirmação via `/metrics` (formato Prometheus)
  de que o tráfego real estava sendo medido e contabilizado.
- O componente de **Pipeline Batch** (chamando `/predict/batch`
  periodicamente) permanece descrito nesta arquitetura como a próxima
  peça a ser implementada operacionalmente (ex.: via Cloud Scheduler +
  Cloud Functions no mesmo projeto GCP) — fora do escopo de validação
  desta fase, mas tecnicamente direto de adicionar dado que o endpoint já
  existe e está em produção.
- Um **dashboard Streamlit** (`app/streamlit_app.py`) foi construído como
  cliente visual da API, oferecendo tanto a consulta pontual (aba
  "Cliente único", via `/infer`) quanto a priorização de carteira (aba
  "Carteira CSV", via `/predict/batch`) em uma interface única — uma
  aproximação prática de como os dois modos de uso desta arquitetura
  híbrida convergem para o mesmo usuário final (time de Retenção/CRM).
  Possui infraestrutura de deploy própria (`Dockerfile.streamlit`),
  independente da API.

Detalhes completos do deploy (comandos `gcloud` usados, evidências de
teste): ver [README.md](../README.md#deploy-em-nuvem-bônus).
