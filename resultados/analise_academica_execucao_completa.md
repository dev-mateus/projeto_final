# Análise Acadêmica dos Resultados - Execução Completa

## 1. Contexto Experimental

A execução completa do pipeline foi realizada com o conjunto integral de dados (284807 transações), validação cruzada estratificada com 5 folds e comparação entre três estratégias de balanceamento (sem balanceamento, SMOTE e undersampling) em três modelos (Dummy, Regressão Logística e SVM-RBF).

O problema apresenta desbalanceamento extremo: 492 fraudes em 284807 registros, com prevalência de fraude de aproximadamente 0,173% ($p \approx 0{,}001727$), equivalente a uma razão próxima de 578:1 entre classe majoritária e minoritária. Esse cenário torna inadequada a interpretação isolada de acurácia.

## 2. Síntese dos Resultados Principais

### 2.1 Ranking por AUC-PR (métrica central para classe rara)

1. Sem balanceamento + SVM-RBF: AUC-PR = 0,8188
2. Sem balanceamento + Regressão Logística: AUC-PR = 0,7607
3. SMOTE + Regressão Logística: AUC-PR = 0,7312
4. Undersampling + Regressão Logística: AUC-PR = 0,6993
5. Undersampling + SVM-RBF: AUC-PR = 0,6747
6. SMOTE + SVM-RBF: AUC-PR = 0,6642

### 2.2 Melhor configuração global

A melhor configuração global, pelo critério primário do pipeline (AUC-PR), foi:

- Cenário: sem balanceamento
- Modelo: SVM-RBF
- Métricas OOF (out-of-fold):
  - Acurácia = 0,999368
  - Precisão = 0,9535
  - Recall = 0,6667
  - F1 = 0,7847
  - MCC = 0,7970
  - Balanced Accuracy = 0,8333
  - G-mean = 0,8165
  - Brier = 0,000496
  - AUC-ROC = 0,9455
  - AUC-PR = 0,8156

Os valores OOF e de média em CV são consistentes entre si, sugerindo estabilidade de estimação.

## 3. Discussão Crítica por Métrica

## 3.1 Por que Acurácia não é suficiente

As configurações Dummy obtiveram acurácia média de aproximadamente 0,9983 mesmo com recall igual a 0 para fraude. Isso confirma o comportamento esperado em problemas altamente desbalanceados: um classificador trivial pode aparentar desempenho elevado ao prever sempre a classe majoritária.

## 3.2 Interpretação de AUC-PR, Precisão e Recall

Para detecção de fraude, AUC-PR é mais informativa que AUC-ROC, pois enfatiza desempenho na classe positiva rara. Nesse critério, sem balanceamento + SVM-RBF superou as demais combinações.

Observa-se um trade-off claro:

- Com balanceamento (SMOTE e undersampling), o recall aumenta substancialmente (até cerca de 0,915), mas a precisão cai acentuadamente (em torno de 0,048 a 0,101), indicando aumento de falsos positivos.
- Sem balanceamento, o SVM-RBF mantém precisão muito alta (0,954) com recall moderado (0,667), o que elevou F1, MCC e AUC-PR.

Em termos práticos, para ambientes com alto custo operacional de investigação de alertas, a configuração sem balanceamento + SVM-RBF mostra melhor compromisso global.

## 3.3 Métricas adicionais para desbalanceamento (MCC, G-mean, Balanced Accuracy, F-beta)

As métricas adicionais reforçam a mesma conclusão:

- MCC máximo em sem balanceamento + SVM-RBF (0,797), indicando melhor correlação entre predições e rótulos reais quando se considera a matriz de confusão completa.
- G-mean e Balanced Accuracy também favorecem essa configuração frente às alternativas com oversampling/undersampling.
- O F-beta com $\beta = 2$ não altera o ranking principal, apesar de premiar recall.

Esses resultados atendem à diretriz metodológica de incluir métricas além das tradicionais e mostrar que diferentes critérios podem levar a escolhas distintas de modelo.

## 4. Estabilidade Estatística (média e desvio-padrão em CV)

Os desvios-padrão das métricas da melhor configuração são baixos (por exemplo, desvio-padrão da AUC-PR em torno de 0,0305 para sem balanceamento + SVM-RBF), sugerindo consistência entre folds e risco moderado de variação amostral.

Por outro lado, métodos com balanceamento apresentam maior oscilação em algumas métricas, especialmente quando a precisão é muito baixa, refletindo sensibilidade ao particionamento em dados raros.

## 5. Conclusão Acadêmica

A análise multicaso indica que a melhor decisão, sob critério prioritário de AUC-PR e suporte de métricas robustas para desbalanceamento, é a configuração sem balanceamento com SVM-RBF. Embora SMOTE e undersampling elevem recall, o custo em precisão e MCC reduz a qualidade global da triagem.

Assim, para este dataset e este desenho experimental, a evidência empírica sustenta que o balanceamento explícito não foi necessário para obter melhor desempenho geral na classe fraudulenta. A escolha final é tecnicamente justificável por convergência de métricas (AUC-PR, F1, MCC, G-mean e Balanced Accuracy) e pela estabilidade observada na validação cruzada.

## 6. Limitações e Próximos Passos

1. O critério de decisão foi baseado em limiar padrão de classificação; análises orientadas a custo podem beneficiar-se de ajuste de threshold para maximizar utilidade operacional.
2. Avaliar calibração e custo de falsos positivos/falsos negativos de forma explícita pode refinar a decisão de implantação.
3. Repetições adicionais de CV (ou avaliação temporal, quando aplicável) podem fortalecer inferência sobre generalização fora da amostra.
