# Plano de Monitoramento

Este plano cobre duas camadas de observabilidade, já refletidas na
separação de endpoints da API (Etapa 3): **saúde operacional da API**
(`/metrics`, formato Prometheus) e **qualidade do modelo** (`/metadata`,
métricas vindas de `model_metadata.json`).

## 1. Métricas monitoradas

### 1.1 Operacionais (infraestrutura/API)

Já instrumentadas em `src/churn_prediction/metrics.py` e expostas em
`GET /metrics`:

| Métrica | Tipo Prometheus | O que indica |
|---|---|---|
| `churn_api_requests_total` | Counter (por `method`, `path`, `status_code`) | Volume de tráfego e taxa de erro por rota |
| `churn_api_request_latency_seconds` | Histogram (por `method`, `path`) | Distribuição de latência — permite calcular p50/p95/p99 |
| `churn_predictions_total` | Counter (por `risk_level`) | Volume e distribuição de predições por nível de risco |

Adicionalmente, todo log estruturado (JSON) inclui `request_id`,
`latency_ms` e `status_code` — consultável diretamente caso ainda não
haja um stack de observabilidade (Prometheus/Grafana) disponível.

### 1.2 Qualidade do modelo

Capturadas no treino (`model_metadata.json`, exposta via `GET /metadata`)
e que devem ser **recalculadas periodicamente** contra dados reais com
rótulo conhecido (ver Seção 3):

| Métrica | Valor de referência (treino atual) | Threshold de alerta sugerido |
|---|---|---|
| AUC-ROC | 0.844 | Alerta se cair abaixo de 0.75 |
| Recall | 0.810 | Alerta se cair abaixo de 0.65 |
| Precisão | 0.509 | Alerta se cair abaixo de 0.35 |
| Custo líquido de negócio | −R$ 191.674,00 | Alerta se o custo líquido se tornar positivo (modelo destruindo valor) |

### 1.3 Distribuição de dados (proxy de *data drift*)

Sem rótulos reais disponíveis em tempo real (o resultado de churn de um
cliente só é conhecido meses depois), o sinal mais prático e barato de
"o modelo pode estar degradando" é monitorar se **a distribuição dos
dados de entrada** se afasta do que foi visto em treino:

| Sinal | Como medir | O que indicaria |
|---|---|---|
| Distribuição de `risk_level` ao longo do tempo | `churn_predictions_total{risk_level=...}` (já instrumentado) | Um salto súbito na proporção de `high` pode indicar drift na entrada, não necessariamente um aumento real de risco |
| Distribuição de `tenure`, `MonthlyCharges` nas requisições recebidas | Logging adicional (não implementado nesta fase — ver Seção 5) | Mudança na composição da base de clientes (ex.: nova campanha de aquisição muda o perfil médio) |
| Taxa de rejeição HTTP 422 | `churn_api_requests_total{status_code="422"}` | Aumento pode indicar uma mudança upstream nos dados (ex.: novo método de pagamento sendo oferecido, ainda não suportado pelo schema) |

## 2. Alertas propostos

| Alerta | Condição | Severidade | Canal sugerido |
|---|---|---|---|
| API fora do ar | `GET /health` falha ou não responde por > 2 min | Crítico | PagerDuty/on-call |
| Modelo não carregado | `GET /ready` retorna `not_ready` por > 5 min | Crítico | PagerDuty/on-call |
| Latência degradada | p95 de `churn_api_request_latency_seconds{path="/infer"}` > 300ms (SLO do ML Canvas) por > 10 min | Alto | Slack do time de Plataforma/ML |
| Taxa de erro elevada | Proporção de `status_code` 5xx > 1% das requisições em 15 min | Alto | Slack do time de Plataforma/ML |
| Taxa de rejeição de entrada elevada | Proporção de `status_code` 422 > 5% das requisições em 1h | Médio | Slack do time de Dados/Integração (provável causa upstream) |
| Distribuição de risco anômala | Proporção de `risk_level=high` varia > 2x em relação à média móvel dos últimos 7 dias | Médio | Slack do time de Ciência de Dados (investigar drift) |
| Degradação de qualidade do modelo | Re-avaliação periódica (Seção 3) mostra AUC-ROC < 0.75 ou recall < 0.65 | Alto | Time de Ciência de Dados + decisão de re-treino |
| Custo de negócio negativo (modelo destruindo valor) | Re-avaliação periódica mostra custo líquido positivo | Crítico | Escalar para a diretoria/stakeholder de negócio |

## 3. Re-avaliação periódica da qualidade do modelo

Como o rótulo real de churn só é conhecido com atraso (um cliente "ainda
não cancelou" não é o mesmo que "não vai cancelar"), a re-avaliação
formal das métricas de qualidade (Seção 1.2) segue uma cadência diferida:

1. **Mensal**: comparar as predições feitas há ~60-90 dias contra o status
   real de churn observado desde então (dado suficiente para a maioria
   dos clientes já ter decidido cancelar ou não, dentro da janela
   considerada no framework de custo do ML Canvas — 12 meses de receita).
2. Recalcular AUC-ROC, recall, precisão e o custo de negócio líquido sobre
   essa amostra "madura".
3. Registrar o resultado como uma nova run no MLflow (mesmo experimento
   `churn-prediction`), permitindo comparação histórica da qualidade do
   modelo em produção ao longo do tempo — não apenas no momento do treino.
4. Se qualquer threshold de alerta da Seção 2 for violado, abrir o
   playbook de resposta (Seção 4).

## 4. Playbook de resposta a incidentes

### 4.1 API fora do ar / modelo não carregado

1. Verificar `GET /health` e `GET /ready` diretamente.
2. Se `/health` falha: problema de infraestrutura (processo caiu, recurso
   esgotado) — reiniciar o serviço; verificar logs estruturados mais
   recentes para a causa raiz (ex.: `OOM`, erro de inicialização).
3. Se `/health` ok mas `/ready` reporta `not_ready`: os artefatos do
   modelo (`models/*.pt`, `*.joblib`) não foram encontrados ou falharam
   ao carregar — verificar se o volume/storage onde os artefatos residem
   está acessível; re-executar `make train` se os artefatos realmente
   estiverem ausentes.
4. Comunicar o stakeholder (time de Retenção/CRM) se o tempo de
   indisponibilidade for relevante, já que decisões de campanha podem
   depender do score atualizado.

### 4.2 Degradação de latência

1. Consultar `GET /metrics` para confirmar em qual rota a latência
   degradou (`/infer` vs. `/predict/batch` têm perfis de custo muito
   diferentes).
2. Verificar se há um pico de volume correlato em
   `churn_api_requests_total` — degradação pode ser simplesmente
   throughput acima do dimensionado, não um bug.
3. Se for `/predict/batch`: confirmar se lotes estão próximos do limite
   máximo (500 clientes) com frequência incomum — pode justificar revisar
   o limite ou paralelizar o processamento interno.
4. Se persistir sem explicação de volume: escalar para investigação de
   performance (profiling do `forward` da MLP, contenção de CPU no host).

### 4.3 Degradação de qualidade do modelo (re-avaliação periódica)

1. Confirmar que a degradação não é um artefato de mudança no pipeline de
   dados upstream (ex.: uma coluna começou a vir com formato diferente) —
   checar a suíte de testes de schema (`tests/test_schema.py`) contra uma
   amostra recente de produção.
2. Investigar se a composição da base de clientes mudou de forma
   significativa (nova campanha de aquisição, mudança de portfólio de
   planos) — sinal de *data drift* real, não necessariamente um defeito
   do modelo.
3. Se confirmado data drift ou degradação genuína: iniciar re-treinamento
   com dados mais recentes, seguindo o mesmo pipeline reprodutível
   (`make train`), e comparar a nova run contra a anterior no MLflow antes
   de promover o modelo novo a produção.
4. Documentar a decisão (mesmo que seja "manter o modelo atual") e
   atualizar o [Model Card](model_card.md) com a data e o resultado da
   reavaliação.

### 4.4 Custo de negócio líquido se tornando positivo

Esse é o cenário mais grave do ponto de vista de negócio — o modelo
passaria a destruir valor em vez de gerar. Resposta imediata sugerida:

1. Considerar reverter temporariamente para o baseline mais simples
   (Regressão Logística, Etapa 2) ou desativar a priorização automática
   até a causa ser entendida — o framework de custo já demonstrou que
   mesmo um modelo mais simples supera com folga o `DummyClassifier`.
2. Investigar se o custo de campanha real (`campaign_cost`) mudou de
   forma que o threshold ou os pesos do modelo deixaram de ser
   adequados — revisitar a análise de sensibilidade ao custo (Etapa 2).
3. Escalar para o stakeholder de negócio (diretoria) antes de qualquer
   decisão definitiva, dado o impacto financeiro direto.

## 5. Lacunas conhecidas deste plano (próximos passos, fora do escopo desta fase)

- Não há, nesta fase, um job automatizado de re-avaliação mensal (Seção
  3) — o processo está documentado, mas a execução ainda seria manual.
- Não há monitoramento automatizado de distribuição de features de
  entrada (Seção 1.3) — apenas a proposta de quais sinais seriam úteis.
- Não há integração configurada com uma ferramenta real de alerta
  (PagerDuty, Slack) — os canais sugeridos na Seção 2 são uma proposta de
  design, a ser conectada na infraestrutura real de quem operar este
  modelo em produção.
