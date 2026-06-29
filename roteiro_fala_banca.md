# Roteiro de Fala para Banca (Mateus e Pedro)

## 1) Como usar este roteiro

Este material foi feito para estudo e treino de apresentação.

Estrutura de cada bloco:

- O que falar: sugestão de fala em linguagem natural.
- O que precisa ficar claro para a banca: mensagem principal do slide.
- Tradução simples: versão não técnica para fixar o conceito.
- Possíveis perguntas da banca: perguntas prováveis e respostas curtas.

Sugestão de divisão entre os apresentadores:

- Mateus: metodologia experimental + explicação do código + resultados quantitativos.
- Pedro: contexto, revisão bibliográfica, discussão crítica, limitações e conclusão.
- Ambos: perguntas finais e defesa de escolhas metodológicas.

---

## 2) O que vocês precisam dominar antes de apresentar (checklist)

### 2.1 Núcleo técnico mínimo

- Saber explicar por que acurácia sozinha é enganosa quando fraude é 0,17%.
- Saber diferenciar precisão e recall em termos de impacto no banco.
- Saber por que AUC-PR foi priorizada em vez de AUC-ROC.
- Saber justificar por que o melhor modelo foi SVM-RBF sem rebalanceamento.
- Saber explicar o trade-off: mais recall com SMOTE/undersampling, mas muitos falsos positivos.

### 2.2 Núcleo de código mínimo

- Entender o fluxo do arquivo fraud_detection_pipeline.py de ponta a ponta.
- Entender como o pipeline evita vazamento de dados (balanceamento só no treino do fold).
- Entender como o melhor modelo é escolhido (ordem: AUC-PR, recall, MCC).
- Entender os artefatos gerados (CSV e figuras) e de onde vêm os números dos slides.

### 2.3 Núcleo de rigor acadêmico

- Dizer claramente o que foi implementado e o que foi discutido conceitualmente.
- Exemplo importante: DCA e perda esperada aparecem na discussão do artigo, mas não estão implementadas no pipeline principal.
- Assumir limitações: não houve teste formal de hipótese entre modelos.

---

## 3) Roteiro de fala por slide (apresentacao_banca_marp.md)

## Slide 1 - Título

### O que falar

"Boa tarde. Nós somos Mateus e Pedro, e vamos apresentar nosso projeto sobre detecção de fraudes em cartões de crédito com aprendizado de máquina. O foco do trabalho foi comparar modelos e métricas em um cenário extremamente desbalanceado, que é o que acontece na prática em dados financeiros reais."

### O que precisa ficar claro para a banca

- Tema relevante e aplicado.
- Problema real: classe fraudulenta muito rara.

### Tradução simples

"É como procurar poucas agulhas em um palheiro enorme, sem errar demais para não atrapalhar o cliente." 

---

## Slide 2 - Roteiro

### O que falar

"Nossa apresentação está organizada em sete partes: contexto, objetivos, revisão, metodologia, resultados, discussão e conclusão. Dessa forma, mostramos desde o problema até a justificativa da decisão final de modelo."

### O que precisa ficar claro para a banca

- Estrutura lógica de artigo científico.

---

## Slide 3 - Contexto do Problema

### O que falar

"No nosso dataset temos 284.807 transações, mas apenas 492 fraudes, ou seja, 0,1727%. Esse desbalanceamento é extremo. Nesse cenário, um modelo que quase sempre responde 'não é fraude' já teria acurácia muito alta, mas seria inútil para detectar fraude de verdade."

### O que precisa ficar claro para a banca

- Por que o problema é tecnicamente difícil.
- Por que acurácia pode enganar.

### Tradução simples

"Se 999 de 1000 casos são normais, alguém que sempre diz 'normal' parece acertar muito, mas não resolve o problema real." 

### Possível pergunta da banca

- Pergunta: "Qual a prevalência exata de fraude?"
- Resposta: "Aproximadamente 0,1727%, com 492 fraudes em 284.807 registros." 

---

## Slide 4 - Objetivo do Trabalho

### O que falar

"Nosso objetivo foi comparar modelos de classificação para detectar fraude, avaliando o efeito de técnicas de balanceamento e de métricas robustas para classe rara. Comparamos Dummy, Regressão Logística e SVM-RBF em três cenários: sem balanceamento, SMOTE e undersampling."

### O que precisa ficar claro para a banca

- Objetivo é comparativo e orientado à decisão prática.

---

## Slide 5 - Perguntas de Pesquisa

### O que falar

"Nós queríamos responder quatro perguntas: se SMOTE melhora recall sem destruir precisão, se AUC-PR é mais informativa que AUC-ROC, se métricas de desbalanceamento mudam a escolha do modelo e qual configuração oferece melhor compromisso técnico-operacional."

### O que precisa ficar claro para a banca

- Trabalho guiado por perguntas, não por tentativa aleatória.

---

## Slide 6 - Revisão Bibliográfica e Lacuna

### O que falar

"Na revisão, observamos que muitos estudos relatam ganho estatístico, mas nem sempre traduzem isso para impacto operacional. Nossa contribuição foi justamente conectar desempenho preditivo com custo de falsos alertas."

### O que precisa ficar claro para a banca

- Lacuna clara e contribuição alinhada.

### Tradução simples

"Não basta 'acertar no papel'; precisa funcionar sem travar a operação do banco." 

---

## Slide 7 - Dados e Pré-processamento

### O que falar

"Usamos o dataset público da ULB/Kaggle. No pré-processamento validamos a coluna alvo, otimizamos tipos para reduzir memória e aplicamos padronização para modelos sensíveis à escala."

### O que precisa ficar claro para a banca

- Cuidado com consistência dos dados.
- Eficiência computacional sem alterar o problema.

### Tradução simples

"Foi como organizar e padronizar os documentos antes da análise, para evitar erro e ganhar velocidade." 

---

## Slide 8 - Pipeline Experimental

### O que falar

"Aplicamos validação cruzada estratificada com 5 folds e random_state fixo para reprodutibilidade. Em cada fold, treinamos com um cenário de balanceamento e um modelo, depois agregamos as previsões em Out-of-Fold. Isso permite avaliação mais estável."

### O que precisa ficar claro para a banca

- Método robusto de validação.
- Evita viés de uma única divisão treino-teste.

### Possível pergunta da banca

- Pergunta: "Como evitaram vazamento de dados com SMOTE?"
- Resposta: "O balanceamento é aplicado dentro do pipeline, apenas no treino de cada fold, nunca no conjunto de validação." 

---

## Slide 9 - Métricas de Avaliação

### O que falar

"Usamos um conjunto amplo de métricas: precisão, recall, F1, F-beta, MCC, balanced accuracy, G-mean, Brier, AUC-ROC e AUC-PR. A métrica prioritária para escolha foi AUC-PR, por ser mais adequada à classe rara."

### O que precisa ficar claro para a banca

- Escolha de métrica foi justificada pelo problema.

### Tradução simples

"AUC-PR olha com lupa justamente para os casos de fraude, que são os que importam." 

---

## Slide 10 - Resultados Consolidados (Tabela)

### O que falar

"A melhor configuração foi sem balanceamento com SVM-RBF, com AUC-PR de 0,8188, precisão de 0,9542 e MCC de 0,7972. Regressão logística sem balanceamento ficou em segundo lugar. Os cenários com rebalanceamento subiram recall, mas com queda forte de precisão e MCC."

### O que precisa ficar claro para a banca

- Resultado principal quantitativo.
- Trade-off explícito.

### Possível pergunta da banca

- Pergunta: "Por que não escolher o modelo com maior recall?"
- Resposta: "Porque o custo em falsos positivos ficou muito alto, o que inviabiliza uso operacional." 

---

## Slide 11 - Comparação Visual

### O que falar

"Este gráfico resume o comportamento geral: quando buscamos só sensibilidade, perdemos confiabilidade dos alertas. O SVM sem balanceamento manteve o melhor equilíbrio entre detectar fraude e não gerar alerta falso em excesso."

### O que precisa ficar claro para a banca

- Visual confirma a tabela.

---

## Slide 12 e 13 - Curva Precision-Recall

### O que falar

"Aqui está a curva Precision-Recall da melhor configuração. A área sob essa curva foi alta para um cenário tão desbalanceado. Isso reforça que o modelo separa bem fraudes de não fraudes onde realmente importa."

### O que precisa ficar claro para a banca

- Evidência visual da qualidade do classificador campeão.

### Tradução simples

"É como mostrar que, ao procurar fraude, o modelo continua confiável em boa parte da busca." 

---

## Slide 14 e 15 - Matriz de Confusão

### O que falar

"A matriz de confusão mostra o custo real dos erros. Temos muitos verdadeiros negativos, boa quantidade de fraudes capturadas e poucos falsos positivos comparado às estratégias reamostradas."

### O que precisa ficar claro para a banca

- Erro tipo I e tipo II têm impacto operacional diferente.

### Possível pergunta da banca

- Pergunta: "Qual erro é pior para o banco?"
- Resposta: "Depende da política de risco, mas muitos falsos positivos geram bloqueio indevido e custo operacional; muitos falsos negativos deixam fraude passar." 

---

## Slide 16 - Leitura Operacional FP/TP

### O que falar

"Este slide traduz métrica em operação: com SVM sem balanceamento, temos cerca de 0,05 falso positivo por verdadeiro positivo. Com SMOTE e undersampling na regressão, esse número sobe para 15,89 e 19,85. Isso significa um volume de investigação impraticável."

### O que precisa ficar claro para a banca

- Resultado estatístico foi convertido para impacto de negócio.

### Tradução simples

"É a diferença entre uma fila curta de alertas úteis e uma fila gigante de alertas falsos." 

---

## Slide 17 - Discussão Crítica

### O que falar

"Nossa discussão central é: em classe rara, acurácia e até ROC podem parecer boas, mas não bastam para decisão. AUC-PR e MCC foram mais consistentes para escolher modelo com utilidade real."

### O que precisa ficar claro para a banca

- Maturidade crítica na interpretação dos resultados.

---

## Slide 18 - Limitações

### O que falar

"Nosso estudo tem limitações: base histórica estática, ausência de teste formal de hipótese entre modelos e custo computacional elevado em algumas combinações, como SVM com reamostragem. Além disso, DCA e perda esperada ficaram no plano conceitual, não no pipeline principal."

### O que precisa ficar claro para a banca

- Transparência metodológica.

### Possível pergunta da banca

- Pergunta: "Por que não fizeram DCA no código final?"
- Resposta: "Priorizamos consolidar a comparação robusta de métricas e os cenários de balanceamento; DCA foi mantida como extensão futura planejada." 

---

## Slide 19 - Conclusões

### O que falar

"Concluímos que o melhor compromisso técnico-operacional foi sem balanceamento com SVM-RBF. O estudo reforça que escolher a métrica correta é tão importante quanto escolher o algoritmo."

### O que precisa ficar claro para a banca

- Conclusão é consistente com objetivo e evidências.

---

## Slide 20 - Trabalhos Futuros

### O que falar

"Como continuidade, sugerimos avaliação temporal para drift, ajuste de limiar orientado a custo, testes inferenciais entre modelos e investigação de arquiteturas mais escaláveis para produção."

### O que precisa ficar claro para a banca

- Trabalho abre agenda de pesquisa concreta.

---

## Slide 21 - Contribuições

### O que falar

"Do ponto de vista acadêmico, entregamos comparação robusta em cenário de desbalanceamento extremo. Do ponto de vista prático, entregamos pipeline reprodutível e critérios objetivos para escolha de modelo com impacto operacional." 

---

## Slide 22 - Encerramento

### O que falar

"Obrigado. Estamos à disposição para perguntas."

---

## 4) Explicação completa do código (fraud_detection_pipeline.py)

## 4.1 Visão geral em linguagem simples

Pensem no código como uma linha de produção com 6 estações:

1. Ler os dados.
2. Fazer uma checagem inicial e gerar gráficos básicos.
3. Testar combinações de modelos e balanceamentos com validação cruzada.
4. Salvar tabelas de métricas.
5. Gerar gráficos de comparação.
6. Escolher o melhor modelo e gerar gráficos detalhados dele.

---

## 4.2 Bloco de configuração e entrada (CLI)

### O que o código faz

- Define parâmetros como caminho dos dados, número de folds, uso de SVM, modo rápido e número de núcleos de CPU.
- Permite repetir o experimento com diferentes configurações sem editar o código.

### Tradução simples

"É o painel de controle do experimento." 

### Pontos-chave para falar

- random_state fixo (42) para reprodutibilidade.
- --n-jobs controla paralelismo.
- --fast-mode existe para iteração local (não foi o modo principal do artigo final).

---

## 4.3 Carregamento e otimização de dados

### O que o código faz

- Lê o CSV.
- Verifica se a coluna Class existe.
- Converte tipos para reduzir memória (`float32`, `int8`).

### Tradução simples

"Antes de analisar, arrumamos a planilha para ela ficar mais leve e confiável." 

---

## 4.4 EDA automática

### O que o código faz

- Calcula resumo de classes e dados faltantes.
- Salva estatísticas de Time e Amount.
- Gera gráficos de distribuição de classes, Time e Amount.

### Tradução simples

"É o diagnóstico inicial para conhecer o perfil dos dados." 

---

## 4.5 Catálogo de modelos e cenários

### O que o código faz

- Modelos principais:
  - DummyClassifier (baseline)
  - LogisticRegression
  - SVC RBF
- Cenários:
  - sem_balanceamento
  - smote
  - undersampling
- Random Forest existe como opcional.

### Tradução simples

"Montamos um cardápio de estratégias para testar de forma justa." 

---

## 4.6 Montagem do pipeline por fold

### O que o código faz

A função build_pipeline junta etapas na ordem:

1. scaler (quando necessário)
2. sampler (se houver)
3. classificador

Isso garante que cada configuração seja treinada da mesma forma estrutural.

### Por que isso é importante

- Evita inconsistência entre experimentos.
- Ajuda a evitar vazamento quando usado dentro do fold.

---

## 4.7 Processamento de cada fold (_process_fold)

### O que o código faz

- Recebe índices de treino/teste daquele fold.
- Clona modelo e sampler para não contaminar execuções.
- Treina no treino.
- Prediz no teste.
- Calcula métricas daquele fold.

### Tradução simples

"Cada fold é um mini-experimento independente." 

---

## 4.8 Conversão de score (get_scores)

### O que o código faz

- Se o modelo tem predict_proba, usa probabilidade.
- Se tem decision_function, normaliza para [0,1].
- Se não tem nenhum, usa predição binária.

### Por que isso importa

- Permite calcular AUC e Brier de forma consistente entre modelos.

---

## 4.9 Métricas (compute_metrics)

### O que o código calcula

- accuracy, precision, recall, f1
- mcc, balanced_accuracy, gmean, fbeta_2
- brier, roc_auc, pr_auc

### Tradução simples

"Em vez de uma nota única, o modelo recebe um boletim completo." 

---

## 4.10 Orquestração da validação cruzada (run_cv_experiments)

### O que o código faz

- Cria os folds estratificados.
- Para cada cenário e cada modelo:
  - roda folds em paralelo
  - guarda métricas por fold
  - concatena previsões OOF
- Gera:
  - dataframe por fold
  - resumo média/desvio-padrão
  - dicionário com OOF por configuração

### Tradução simples

"É a fábrica principal onde todas as combinações são testadas de forma padronizada." 

---

## 4.11 Escolha do melhor modelo (choose_best_config)

### Regra usada

Ordenação por:

1. maior pr_auc_mean
2. maior recall_mean
3. maior mcc_mean

### Tradução simples

"Primeiro escolhemos quem melhor encontra fraude com qualidade; em empate, priorizamos capturar mais fraudes e depois equilíbrio geral." 

---

## 4.12 Geração de saídas

### Tabelas (CSV)

- metricas_por_fold.csv
- metricas_resumo_media_desvio.csv
- comparacao_acuracia_vs_metricas_imbalance.csv
- melhor_config_metricas_oof.csv

### Figuras

- comparacao_metricas_principais.png
- melhor_config_curva_roc.png
- melhor_config_curva_pr.png
- melhor_config_curva_calibracao.png
- melhor_config_matriz_confusao.png
- EDA (classe, time, amount)

### Tradução simples

"O código não só calcula: ele gera evidências prontas para o artigo e para a banca." 

---

## 5) Termos técnicos traduzidos para linguagem não técnica

- Classe desbalanceada: quando quase tudo é de um tipo e o tipo importante é raro.
- Baseline Dummy: modelo mais simples possível, usado como referência mínima.
- Fold: uma divisão dos dados para treino/teste repetido.
- Out-of-Fold: juntar as predições de todos os testes dos folds para avaliar geral.
- Recall: quanto das fraudes reais o modelo conseguiu pegar.
- Precisão: dos alertas emitidos, quantos eram fraude de verdade.
- MCC: nota de equilíbrio geral que considera todos os acertos e erros.
- AUC-PR: qualidade do modelo focando na classe rara.
- AUC-ROC: qualidade global, mas pode parecer melhor do que realmente é em classe rara.
- Brier score: qualidade da probabilidade prevista (calibração).

---

## 6) Analogias úteis para usar na apresentação

- Fraude em base desbalanceada:
  - "Procurar agulha no palheiro."
- Precisão vs recall:
  - "Recall é rede larga para não deixar peixe escapar; precisão é qualidade do peixe capturado."
- Escolha de métrica:
  - "Acurácia é olhar a média da turma; AUC-PR é olhar especificamente quem estava em risco de reprovar."
- OOF:
  - "É como várias provas parciais que viram uma nota final mais justa que uma prova única."

---

## 7) Perguntas difíceis da banca e resposta sugerida

## 7.1 "Por que SVM sem balanceamento foi melhor que SMOTE?"

Resposta curta:

"Porque no nosso cenário o ganho de recall com SMOTE veio com queda drástica de precisão e MCC. Em operação, isso geraria muitos falsos positivos. O SVM sem balanceamento preservou melhor equilíbrio técnico-operacional." 

## 7.2 "Por que não usar só ROC-AUC?"

Resposta curta:

"Com classe rara, ROC-AUC pode ficar otimista por causa do grande número de verdadeiros negativos. AUC-PR foca na classe positiva, que é nosso alvo principal." 

## 7.3 "Cadê DCA e perda esperada no código?"

Resposta curta:

"Esses itens foram discutidos como direção de maturidade analítica, mas não implementados no pipeline principal desta versão. Mantivemos transparência metodológica sobre isso." 

## 7.4 "Como garantem que não houve vazamento?"

Resposta curta:

"Balanceamento e escalonamento ocorrem dentro do pipeline em cada fold, aplicados ao treino daquela partição, nunca ao teste." 

## 7.5 "Como reproduzir os resultados?"

Resposta curta:

"Executando o fraud_detection_pipeline.py com dataset creditcard.csv e parâmetros padrão. O código gera automaticamente tabelas e figuras usadas no artigo e nos slides." 

---

## 8) Treino final (simulação entre vocês)

## 8.1 Treino rápido de 15 minutos

- 5 min: Mateus (metodologia + código)
- 5 min: Pedro (resultados + discussão)
- 5 min: perguntas cruzadas entre vocês

## 8.2 Treino completo de 30 a 40 minutos

- Apresentação inteira com cronômetro
- Gravar áudio
- Revisar 3 pontos:
  - clareza sem ler texto
  - domínio dos números principais
  - consistência entre artigo, slides e código

---

## 9) Números-chave para decorar

- Total de transações: **284.807**
- Fraudes: **492**
- Prevalência: **0,1727%**
- Melhor configuração: **sem balanceamento + SVM-RBF**
- AUC-PR (CV): **0,8188**
- AUC-PR (OOF): **0,8156**
- Precisão (melhor): **0,9535**
- Recall (melhor): **0,6667**
- MCC (melhor): **0,7970**
- FP/TP (melhor): **0,05**

---

## 10) Fechamento sugerido para a banca

"Nosso principal resultado foi mostrar que, em desbalanceamento extremo, a escolha da métrica muda a decisão final de modelo. O SVM-RBF sem rebalanceamento foi o melhor compromisso entre detectar fraudes e manter alertas confiáveis. Além do resultado técnico, entregamos um pipeline reprodutível e uma análise com foco no impacto operacional real." 
# Roteiro Completo de Fala para a Banca

## Projeto
Detecção de Fraudes em Cartões de Crédito com Aprendizado de Máquina

## Objetivo deste roteiro
Este documento foi feito para você (Mateus) e o Pedro estudarem antes da apresentação.
Ele traz:

1. O que precisa estar entendido antes de apresentar.
2. O que falar em cada slide, em linguagem clara.
3. Tradução de termos técnicos para termos não técnicos.
4. Explicação completa do código principal do projeto.
5. Perguntas prováveis da banca e respostas sugeridas.

---

## 1) O que precisa ser entendido antes da apresentação

### 1.1 A ideia central (em uma frase)
Em problema com fraude rara, acertar quase tudo não é difícil; difícil é acertar a fraude sem gerar alerta falso demais.

### 1.2 Conceitos que vocês precisam dominar

1. **Desbalanceamento extremo**
   - Técnico: a classe positiva (fraude) é muito menor que a negativa.
   - Não técnico: é como procurar agulha em um palheiro.

2. **Paradoxo da acurácia**
   - Técnico: acurácia pode ficar alta mesmo sem detectar fraude.
   - Não técnico: se eu disser “ninguém é fraudador”, acerto quase tudo porque fraude é rara.

3. **Precisão vs Recall**
   - Precisão: dos alertas que o modelo gera, quantos são fraude de verdade.
   - Recall: de todas as fraudes reais, quantas o modelo conseguiu capturar.
   - Não técnico: precisão = “qualidade dos alarmes”; recall = “quantidade de fraudes capturadas”.

4. **AUC-PR e AUC-ROC**
   - AUC-ROC pode parecer boa demais quando há muitas transações legítimas.
   - AUC-PR é mais honesta em classe rara.
   - Não técnico: ROC olha o jogo inteiro; PR olha exatamente onde dói (fraudes).

5. **MCC**
   - Técnico: mede correlação entre predição e realidade usando toda a matriz de confusão.
   - Não técnico: nota de equilíbrio geral, sem “maquiagem” de classe majoritária.

6. **Trade-off operacional**
   - Técnico: aumentar recall pode reduzir precisão e elevar falsos positivos.
   - Não técnico: pegar mais fraudes pode significar incomodar muito cliente legítimo.

### 1.3 Mensagem final que a banca deve levar
A melhor configuração não foi a que “caça mais fraude” isoladamente, e sim a que combina boa detecção com alarmes confiáveis e custo operacional viável.

---

## 2) Roteiro de fala slide a slide

## Slide 1 - Título
### Quem fala
Mateus (abertura)

### O que falar
“Bom dia/boa tarde, somos Mateus e Pedro. Nosso trabalho analisa detecção de fraude em cartão de crédito usando aprendizado de máquina em cenário de desbalanceamento extremo. O foco não foi só acertar estatisticamente, mas escolher um modelo que também faça sentido operacionalmente.”

---

## Slide 2 - Roteiro
### Quem fala
Pedro

### O que falar
“Vamos passar por contexto, objetivos, revisão, metodologia, resultados, discussão crítica e conclusões. Ao final, mostramos limitações e próximos passos.”

---

## Slide 3 - Contexto do problema
### Quem fala
Mateus

### O que falar
“Temos 284.807 transações e apenas 492 fraudes, prevalência de 0,1727%. Isso significa que fraude é evento raro. Nesse tipo de dado, acurácia sozinha engana, porque o modelo pode acertar muito simplesmente por prever sempre a classe legítima.”

### Frase de impacto
“Nosso problema não é acertar o óbvio, é detectar o raro com confiança.”

---

## Slide 4 - Objetivo
### Quem fala
Pedro

### O que falar
“O objetivo foi comparar modelos de classificação para fraude sob desbalanceamento extremo, avaliando técnicas de balanceamento e métricas robustas. Testamos Dummy, Regressão Logística e SVM-RBF em três cenários: sem balanceamento, SMOTE e undersampling.”

---

## Slide 5 - Perguntas de pesquisa
### Quem fala
Mateus

### O que falar
“Guiamos o trabalho por quatro perguntas: se SMOTE melhora recall sem destruir precisão; se AUC-PR é mais informativa que AUC-ROC nesse cenário; se métricas robustas mudam a decisão em relação à acurácia; e qual combinação tem melhor compromisso técnico-operacional.”

---

## Slide 6 - Revisão e lacuna
### Quem fala
Pedro

### O que falar
“A revisão mostra muitos trabalhos com foco em métrica estatística global. A lacuna está na tradução desses números para impacto operacional, principalmente falsos alertas. Nosso trabalho preenche isso ao discutir desempenho e custo operacional juntos.”

---

## Slide 7 - Dados e pré-processamento
### Quem fala
Mateus

### O que falar
“Usamos o dataset ULB/Kaggle. A variável alvo é Class. As variáveis V1 a V28 já estão em PCA. No pré-processamento, validamos a coluna alvo, reduzimos tipos numéricos para eficiência e aplicamos padronização para modelos sensíveis à escala.”

### Tradução não técnica
“Foi uma ‘higienização de dados’ para que o treino fosse mais estável e rápido.”

---

## Slide 8 - Pipeline experimental
### Quem fala
Pedro

### O que falar
“Aplicamos validação cruzada estratificada com 5 folds, mantendo a proporção de fraude em cada partição. Em cada fold, treinamos e avaliamos os modelos nos três cenários de balanceamento. O resultado final é agregado em OOF, ou seja, fora da amostra de treino de cada fold.”

### Analogia
“É como testar um aluno em 5 provas diferentes e depois tirar uma média justa.”

---

## Slide 9 - Métricas
### Quem fala
Mateus

### O que falar
“Além de acurácia, usamos precisão, recall, F1, F-beta, MCC, balanced accuracy, G-mean, Brier, AUC-ROC e AUC-PR. Para decisão final, priorizamos AUC-PR por ser mais adequada para classe rara.”

---

## Slide 10 - Tabela de resultados consolidados
### Quem fala
Pedro

### O que falar
“A melhor configuração em AUC-PR foi sem balanceamento com SVM-RBF: 0,8188, com precisão 0,9542, recall 0,6667 e MCC 0,7972. Rebalanceamento aumentou recall, mas derrubou muito a precisão.”

### Leitura didática
“Em linguagem simples: alguns modelos ‘gritam fraude’ demais, mas erram muito alarme.”

---

## Slide 11 - Figura comparativa de métricas
### Quem fala
Mateus

### O que falar
“Essa figura mostra o trade-off de forma visual. Configurações com SMOTE e undersampling sobem em recall, mas perdem qualidade dos alertas. O SVM sem balanceamento ficou mais equilibrado no conjunto de métricas relevantes.”

---

## Slide 12 - Curva PR
### Quem fala
Pedro

### O que falar
“Na curva Precision-Recall da melhor configuração, vemos separação robusta da classe de fraude. A AUC-PR OOF ficou em 0,8156, muito acima da linha de base associada à prevalência.”

### Tradução não técnica
“O modelo está bem acima do ‘chute aleatório’ exatamente na tarefa que importa.”

---

## Slide 13 - Matriz de confusão
### Quem fala
Mateus

### O que falar
“A matriz mostra o comportamento operacional do modelo campeão: poucos falsos positivos em relação ao volume total e captura consistente de fraude. Isso justifica alta precisão com recall moderado-alto.”

---

## Slide 14 - Leitura operacional FP/TP
### Quem fala
Pedro

### O que falar
“Esse é um slide-chave para banca: sem balanceamento + SVM-RBF gera cerca de 0,05 falso positivo por verdadeiro positivo. Já SMOTE + Regressão Logística gera 15,89 e undersampling + Regressão Logística 19,85. Em operação real, isso muda totalmente o custo.”

### Frase de impacto
“Não adianta detectar mais se o sistema vira uma fábrica de alarmes falsos.”

---

## Slide 15 - Discussão crítica
### Quem fala
Mateus

### O que falar
“Confirmamos o paradoxo da acurácia e a necessidade de métricas adequadas à classe rara. AUC-PR e MCC foram mais úteis para decisão. Nossa escolha final foi orientada por equilíbrio entre desempenho preditivo e viabilidade operacional.”

---

## Slide 16 - Limitações
### Quem fala
Pedro

### O que falar
“Limitamos este estudo a dados históricos estáticos, sem teste formal de hipótese entre modelos. Também destacamos que DCA e perda esperada foram discutidos no texto, mas não implementados no pipeline principal.”

---

## Slide 17 - Conclusões
### Quem fala
Mateus

### O que falar
“Concluímos que, para este problema, a melhor configuração foi SVM-RBF sem rebalanceamento explícito. Ela apresentou melhor compromisso entre precisão, AUC-PR e MCC, preservando utilidade operacional.”

---

## Slide 18 - Trabalhos futuros
### Quem fala
Pedro

### O que falar
“Como próximos passos: avaliação temporal para drift, ajuste de limiar orientado a custo, testes estatísticos entre modelos e investigação de abordagens mais escaláveis para produção.”

---

## Slide 19 - Contribuições
### Quem fala
Mateus

### O que falar
“No aspecto acadêmico, mostramos empiricamente o paradoxo da acurácia em classe rara. No aspecto prático, entregamos pipeline reprodutível e critério objetivo de escolha de modelo.”

---

## Slide 20 - Encerramento
### Quem fala
Ambos

### O que falar
“Muito obrigado. Ficamos à disposição para perguntas.”

---

## 3) Explicação completa do código (didática)

Arquivo principal: projeto_final/fraud_detection_pipeline.py

## 3.1 Visão geral da arquitetura
O código é um pipeline de ponta a ponta com 6 macroetapas:

1. Carregar e validar dados.
2. Gerar artefatos de análise exploratória.
3. Rodar validação cruzada para vários modelos e cenários.
4. Salvar tabelas de métricas.
5. Gerar gráficos comparativos.
6. Selecionar melhor configuração e salvar relatório final.

### Analogia geral
Pense no pipeline como uma linha de montagem:
- entrada: dados brutos,
- esteiras: pré-processamento + treino + avaliação,
- saída: tabelas e gráficos prontos para decisão.

---

## 3.2 Bloco de configuração e argumentos

Funções: `parse_args`, `resolve_data_path`, constantes globais.

### O que fazem
- Permitem rodar por linha de comando com parâmetros:
  - `--data`
  - `--output-dir`
  - `--include-rf`
  - `--no-svm`
  - `--n-splits`
  - `--max-rows`
  - `--fast-mode`
  - `--n-jobs`

### Como explicar na banca
“Estruturamos o script para ser reproduzível e flexível. O mesmo código roda em modo completo ou rápido, mantendo rastreabilidade experimental.”

---

## 3.3 Carregamento e preparação dos dados

Funções: `load_data`, `generate_eda_artifacts`.

### O que fazem
- Leem o CSV.
- Verificam se a coluna `Class` existe.
- Reduzem tipos de dados para economizar memória.
- Geram artefatos EDA:
  - resumo em JSON,
  - descritiva de `Time` e `Amount`,
  - gráfico de distribuição de classes,
  - histogramas.

### Tradução não técnica
“Antes de treinar, conferimos a ‘qualidade da matéria-prima’ e tiramos uma radiografia inicial dos dados.”

---

## 3.4 Métricas e funções auxiliares

Funções: `safe_div`, `gmean_from_confusion`, `get_scores`, `compute_metrics`.

### O que fazem
- Evitam divisão por zero.
- Calculam G-mean com base na matriz de confusão.
- Extraem score contínuo do modelo (`predict_proba` ou `decision_function`).
- Calculam todas as métricas usadas no estudo.

### Ponto importante para falar
“Mesmo quando o modelo não devolve probabilidade diretamente, o código converte o score para [0,1], viabilizando AUC e Brier de forma consistente.”

---

## 3.5 Catálogo de modelos e cenários

Funções: `make_model_catalog`, `make_scenario_catalog`.

### O que fazem
- Definem modelos comparados:
  - Dummy,
  - Regressão Logística,
  - SVM-RBF,
  - Random Forest opcional.
- Definem cenários de balanceamento:
  - sem balanceamento,
  - SMOTE,
  - undersampling.

### Mensagem didática
“Padronizamos os experimentos em um catálogo. Isso evita comparação injusta.”

---

## 3.6 Construção do pipeline por configuração

Função: `build_pipeline`.

### O que faz
Monta sequência de etapas para cada experimento:

1. Escalonar (quando necessário).
2. Aplicar amostragem (quando cenário exige).
3. Treinar classificador.

### Por que é correto
A amostragem acontece dentro do fluxo de treino do fold, reduzindo risco de vazamento.

---

## 3.7 Processamento por fold

Função: `_process_fold`.

### O que faz
Para um fold específico:

1. Separa treino e teste.
2. Clona modelo e amostrador (isola estado).
3. Treina pipeline.
4. Prediz classe e score.
5. Calcula métricas do fold.
6. Devolve resultados e predições.

### Analogia
“Cada fold é um mini experimento completo, independente dos demais.”

---

## 3.8 Loop principal de validação cruzada

Função: `run_cv_experiments`.

### O que faz
- Percorre todos os cenários e modelos.
- Cria tarefas de folds.
- Executa em paralelo com joblib.
- Agrega métricas por fold.
- Concatena predições OOF.
- Calcula resumo de média e desvio padrão.

### Ponto forte
“Esse bloco garante robustez estatística e ganho de tempo computacional.”

---

## 3.9 Critério de melhor configuração

Função: `choose_best_config`.

### Regra adotada
Ordenação por:

1. Maior `pr_auc_mean`.
2. Desempate por `recall_mean`.
3. Novo desempate por `mcc_mean`.

### Como explicar
“A prioridade é detectar fraude com qualidade em classe rara, sem ignorar equilíbrio global.”

---

## 3.10 Exportação de resultados

Funções:
- `save_accuracy_vs_pr_table`
- `plot_main_metric_bars`
- `plot_curves_for_best`
- `save_best_config_report`

### O que geram
- CSVs consolidados por fold e por resumo.
- Tabela comparando acurácia com métricas de classe rara.
- Gráficos de barras de métricas.
- Curvas ROC, PR, calibração.
- Matriz de confusão da melhor configuração.
- CSV OOF da melhor configuração.

---

## 3.11 Função principal

Função: `main`.

### Fluxo
1. Lê argumentos.
2. Ajusta comportamento de fast mode.
3. Carrega dados.
4. Gera EDA.
5. Executa CV.
6. Salva tabelas/gráficos.
7. Seleciona melhor configuração.
8. Gera artefatos finais e imprime resumo.

### Mensagem para banca
“Todo o processo é automatizado e reprodutível: mesmos parâmetros, mesmos dados, mesma lógica experimental.”

---

## 4) Dicionário rápido: técnico -> linguagem simples

- Classe minoritária: grupo raro (fraudes).
- Baseline: referência mínima.
- Fold: partição de validação.
- OOF: resultado agregado de previsões fora do treino.
- Overfitting: decorar treino e falhar no real.
- Calibração: probabilidade prevista combinar com frequência real.
- Threshold (limiar): ponto de corte para decidir fraude.

---

## 5) Perguntas prováveis da banca e respostas curtas

## Pergunta 1
Por que não escolheram o modelo com maior recall?

### Resposta
Porque recall alto com precisão muito baixa gera muitos falsos positivos. Em fraude real, isso vira custo operacional e atrito com cliente. Buscamos equilíbrio técnico-operacional.

## Pergunta 2
Por que AUC-PR foi prioritária e não AUC-ROC?

### Resposta
Com classe rara, ROC pode parecer inflada pelos muitos verdadeiros negativos. AUC-PR foca no desempenho da classe positiva, que é o foco do problema.

## Pergunta 3
SMOTE não deveria melhorar tudo?

### Resposta
SMOTE geralmente aumenta recall, mas não garante aumento de precisão. No nosso caso, trouxe sensibilidade maior com custo muito alto em falsos alertas.

## Pergunta 4
Qual a principal limitação?

### Resposta
Dados históricos estáticos e ausência de teste inferencial formal entre modelos. Além disso, algumas combinações podem ter custo computacional elevado.

## Pergunta 5
O que falta para produção?

### Resposta
Definir limiar orientado a custo, monitorar drift, plano de retreino periódico e integração com regras de negócio para tratamento de alertas.

---

## 6) Estratégia de treino entre vocês dois

1. Rodada 1 (20 min): leitura em voz alta, sem se preocupar com tempo.
2. Rodada 2 (20 min): cronometrar cada slide.
3. Rodada 3 (20 min): simular perguntas da banca.
4. Rodada 4 (10 min): ajustar transições entre Mateus e Pedro.

### Meta de tempo
- Apresentação completa: 12 a 18 minutos.
- Reserva para perguntas: 5 a 10 minutos.

---

## 7) Checklist final pré-banca

- Conseguem explicar em 30 segundos por que acurácia engana aqui?
- Conseguem justificar por que AUC-PR foi a métrica de decisão?
- Conseguem explicar o trade-off precisão vs recall com exemplo simples?
- Conseguem descrever o pipeline do código sem ler?
- Conseguem citar as principais limitações com honestidade acadêmica?

Se os cinco itens estiverem claros, vocês estão prontos para a banca.
