# Model Card — Previsão de Churn (MLP)

> Segue o formato proposto por [Mitchell et al. (2019), "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993),
> adaptado ao contexto do Tech Challenge 1 (FIAP — Pós Tech).

## 1. Detalhes do modelo

| Campo | Valor |
|---|---|
| **Nome** | `churn-prediction-mlp` |
| **Versão** | 1.0.0 |
| **Tipo** | Rede neural feedforward (MLP — Multi-Layer Perceptron) |
| **Arquitetura** | 2 camadas ocultas (64 → 32 neurônios), ativação ReLU, Dropout 0.3, 1 saída (logit) |
| **Framework** | PyTorch 2.x |
| **Treinado em** | 2026-06-27 |
| **Seed** | 42 (reprodutibilidade determinística) |
| **Features de entrada** | 19 features brutas → 44 features após pré-processamento (one-hot + escala + `tenure_bucket` derivada) |
| **Licença / uso** | Projeto educacional (Tech Challenge FIAP) — não destinado a uso comercial real sem revalidação |
| **Contato** | Grupo Tech Challenge 1 — FIAP MLE10 (ver README) |

### Resumo não-técnico
Este modelo estima a probabilidade de um cliente de telecomunicações
cancelar o serviço (churn) nos próximos meses, a partir de dados
cadastrais, de contrato e de uso de serviços. O objetivo é permitir que o
time de Retenção priorize contato proativo com os clientes de maior risco.

## 2. Uso pretendido

### Casos de uso primários
- Priorização de campanhas de retenção pelo time de CRM (lista ordenada
  por risco, batch periódico).
- Consulta pontual de risco de um cliente específico durante atendimento
  (via API `/infer`).

### Fora do escopo (uso não pretendido)
- **Decisões automatizadas que afetem o cliente sem revisão humana** (ex.:
  cancelamento automático de benefícios, negativa de crédito) — o modelo
  deve informar uma decisão humana, não substituí-la.
- **Uso em qualquer contexto fora de telecomunicações** sem retreinamento
  — o modelo foi treinado em um dataset específico de uma operadora
  fictícia (Telco Customer Churn, IBM) e não generaliza para outros
  setores ou mesmo outras operadoras com perfil de cliente diferente.
- **Decisões de pricing ou elegibilidade de crédito** — o modelo não foi
  validado para esse fim e usar `SeniorCitizen`/`gender` como insumo para
  esse tipo de decisão levanta questões éticas e, possivelmente, legais.

## 3. Dados de treinamento

- **Fonte**: [Telco Customer Churn (IBM)](https://github.com/IBM/telco-customer-churn-on-icp4d) — dataset público.
- **Volume total**: 7.043 registros. Split: ~80% treino (com 20% adicional
  reservado para validação/early stopping), 20% holdout de teste — todos
  estratificados pela variável-alvo.
- **Taxa de churn base**: 26,54% (desbalanceamento tratado via `pos_weight`
  na função de perda).
- **Período/contexto**: dataset estático, de um único snapshot temporal —
  não captura sazonalidade nem mudanças de mercado ao longo do tempo.

## 4. Métricas de performance (holdout de teste, n=1.409)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.844 |
| PR-AUC (Average Precision) | 0.631 |
| F1-score | 0.625 |
| Precisão | 0.509 |
| **Recall** | **0.810** |
| Acurácia | 0.742 |

**Matriz de confusão (holdout, threshold=0.5):**

| | Previsto: não-churn | Previsto: churn |
|---|---|---|
| **Real: não-churn** | TN = 743 | FP = 292 |
| **Real: churn** | FN = 71 | TP = 303 |

**Leitura**: o modelo foi deliberadamente calibrado (via `pos_weight`) para
priorizar **recall** sobre precisão — prefere-se sinalizar clientes que
não vão cancelar (FP, custo de uma campanha desnecessária) do que deixar
passar um cliente que realmente vai cancelar (FN, perda de receita maior).
Essa escolha está alinhada ao framework de custo de negócio documentado no
[ML Canvas](ml_canvas.md#4-objetivo-de-negócio-business-goal).

### Custo de negócio estimado (holdout de teste)
Usando o framework de custo do ML Canvas (FN = receita anual perdida,
FP = R$50 de custo de campanha):

| Métrica | Valor |
|---|---|
| Custo de FN (71 clientes perdidos sem aviso) | R$ 52.577,40 |
| Custo de FP (292 campanhas desnecessárias) | R$ 14.600,00 |
| Ganho de TP (303 clientes retidos a tempo) | R$ 258.851,40 |
| **Custo líquido (negativo = ganho)** | **−R$ 191.674,00** |

## 5. Performance por subgrupo (análise de fairness)

Avaliação de recall, precisão e AUC-ROC por subgrupo demográfico, no
mesmo holdout de teste, para identificar disparidades de performance.

### Por gênero

| Gênero | n | Recall | Precisão | AUC-ROC |
|---|---|---|---|---|
| Female | 687 | 0.782 | 0.537 | 0.839 |
| Male | 722 | 0.773 | 0.491 | 0.847 |

**Achado**: performance equilibrada entre gêneros — diferença pequena
(≤ 5 p.p. em qualquer métrica), não indicando viés relevante por esse
atributo.

### Por idoso (`SeniorCitizen`)

| Grupo | n | Recall | Precisão | AUC-ROC |
|---|---|---|---|---|
| Não idoso (0) | 1.187 | 0.732 | 0.484 | 0.845 |
| **Idoso (1)** | **222** | **0.908** | **0.597** | **0.775** |

**Achado e limitação**: o modelo tem recall consideravelmente maior para
clientes idosos (90,8% vs. 73,2%), mas AUC-ROC mais baixo (0.775 vs.
0.845) — sugerindo que, embora o modelo acerte mais positivos nesse grupo
no threshold atual, sua capacidade de discriminação geral é menor. Além
disso, o subgrupo é pequeno (222 de 1.409, ~16% do teste), o que torna
essas estimativas mais sensíveis a ruído amostral. **Recomendação**:
monitorar esse subgrupo separadamente em produção antes de confiar
plenamente nessas métricas.

### Por tipo de contrato

| Contrato | n (teste) | Casos reais de churn | Recall |
|---|---|---|---|
| Month-to-month | 773 | 329 | 0.857 |
| One year | 300 | 36 | 0.250 |
| **Two year** | **336** | **9** | **0.000** |

**Limitação importante**: no segmento de contrato de 2 anos, o modelo
**não identificou nenhum dos 9 clientes que efetivamente cancelaram** no
holdout de teste. Esse segmento tem taxa de churn muito baixa (2,7%), o
que significa poucos exemplos positivos disponíveis para o modelo
aprender o padrão — uma limitação estrutural do volume de dados, não um
defeito de implementação. **Em produção, scores de risco para clientes
com contrato de 2 anos devem ser tratados com cautela adicional** — o
modelo está, na prática, pouco testado para identificar churn nesse
segmento específico.

## 6. Limitações conhecidas

1. **Dataset estático e de uma única operadora fictícia.** Não há garantia
   de que os padrões aprendidos generalizem para outras operadoras, países,
   ou para mudanças de mercado (promoções de concorrentes, mudanças
   regulatórias, crises econômicas) não representadas no dataset de 2020.
2. **Baixo recall em segmentos de baixa taxa de churn** (contratos de 1-2
   anos), conforme detalhado na Seção 5 — risco de excesso de confiança em
   scores baixos para esses clientes.
3. **Threshold fixo (0.5) pode não ser ideal para todos os contextos de
   uso.** O framework de custo de negócio (Seção 4) sugere que o
   threshold poderia ser otimizado para o custo real de campanha da
   operadora — caso esse custo real seja muito diferente da estimativa de
   R$50 usada aqui, o threshold ótimo mudaria.
4. **Sem dados de série temporal.** O modelo não usa tendências de uso
   recente (ex.: queda de consumo nos últimos 3 meses) — apenas um
   snapshot estático de cada cliente, o que pode limitar a antecipação de
   churn iminente.
5. **Modelo não interpretável nativamente.** Diferente da Regressão
   Logística (Etapa 2), a MLP não oferece coeficientes diretamente
   interpretáveis — qualquer explicação de "por que este cliente tem risco
   alto" exigiria uma camada adicional (ex.: SHAP), não implementada nesta
   fase.
6. **Validação única, sem re-checagem periódica.** As métricas reportadas
   refletem o desempenho no momento do treino (2026-06-27). Sem
   monitoramento contínuo (ver [plano de monitoramento](monitoring_plan.md)),
   a degradação de performance ao longo do tempo (data drift) não seria
   detectada automaticamente.

## 7. Vieses e considerações éticas

- **`SeniorCitizen` e `gender` são usados como features de entrada.**
  Embora a análise de fairness (Seção 5) não tenha encontrado disparidade
  relevante por gênero, o uso de atributos demográficos sensíveis em
  decisões que afetam o tratamento de um cliente (mesmo que "apenas" uma
  oferta de retenção) merece avaliação contínua, especialmente se a
  empresa decidir usar o score para decisões de maior impacto (ex.:
  elegibilidade de desconto).
- **Causalidade vs. correlação**: o modelo identifica correlações entre
  perfil de cliente e churn (ex.: contrato mês-a-mês, fibra óptica,
  pagamento por cheque eletrônico), mas **não implica causalidade**. Por
  exemplo, "forma de pagamento" pode ser um proxy de outros fatores
  socioeconômicos não capturados no dataset — usar esse sinal para
  decisões que afetem o cliente sem entender o mecanismo causal por trás
  é um risco.
- **Risco de retroalimentação**: se o modelo influenciar quem recebe
  ofertas de retenção, e essas ofertas alteram o comportamento futuro de
  churn, dados de treino futuros estarão "contaminados" pela própria ação
  do modelo — um re-treinamento ingênuo no futuro poderia amplificar
  padrões artificiais. Recomenda-se considerar esse efeito ao planejar
  re-treinamentos periódicos.

## 8. Cenários de falha

| Cenário | Comportamento esperado | Mitigação implementada |
|---|---|---|
| Artefatos do modelo (`models/*.pt`, `*.joblib`) ausentes | API sobe normalmente, mas `/ready` reporta `not_ready` e `/infer`/`/predict` retornam HTTP 503 | Implementado (ver `inference.py`, `api.py`) |
| Entrada com categoria fora do domínio conhecido (ex.: novo método de pagamento) | Requisição rejeitada com HTTP 422 antes de chegar ao modelo | Implementado (validação Pydantic com `Literal`) |
| Exceção inesperada durante a inferência | HTTP 500 com mensagem genérica; detalhes completos apenas no log estruturado | Implementado (`exception_handler` genérico) |
| Lote de predição acima do limite (>500 clientes) | Requisição rejeitada com HTTP 422 | Implementado (`max_length=500` no schema) |
| Data drift (mudança na distribuição de clientes ao longo do tempo) | **Não detectado automaticamente** | Não implementado nesta fase — ver [plano de monitoramento](monitoring_plan.md) para proposta |
| Cliente com perfil totalmente fora da distribuição de treino (ex.: `tenure` no limite máximo de 120, custo `MonthlyCharges` muito acima do observado) | O modelo ainda retorna uma predição (não há detecção de out-of-distribution), podendo ser pouco confiável | Não implementado — risco residual aceito nesta fase |

## 9. Recomendações para uso responsável

1. Scores do modelo devem **apoiar**, não substituir, a decisão humana do
   time de Retenção.
2. Re-treinar e revalidar este Model Card sempre que houver mudança
   relevante na base de clientes ou nas ofertas comerciais da operadora.
3. Monitorar separadamente a performance para o subgrupo de clientes
   idosos e para contratos de longo prazo (Seção 5), dado o volume
   reduzido de exemplos de treino nesses segmentos.
4. Revisitar o threshold de decisão (atualmente 0.5) com o custo real de
   campanha da operadora, não a estimativa de R$50 usada nesta fase.
5. Não usar este modelo, sem adaptação e revalidação extensiva, para
   decisões de pricing, crédito ou qualquer contexto de maior impacto ao
   cliente além de priorização de contato de retenção.
