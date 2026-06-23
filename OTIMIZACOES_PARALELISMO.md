# Otimizações de Paralelismo - fraud_detection_pipeline.py

## Resumo das Mudanças

O pipeline agora **usa todos os núcleos do processador** para paralelizar a execução dos folds na validação cruzada, mantendo os mesmos resultados.

### O que foi alterado

1. **Adição de joblib.Parallel** (import adicionado)
   - Usa o backend `LokyBackend` por padrão (mais estável em Windows)

2. **Nova função `_process_fold()`**
   - Encapsula o processamento de um único fold
   - Pode ser executada em paralelo sem efeitos colaterais

3. **Refatoração de `run_cv_experiments()`**
   - Novo parâmetro: `n_jobs: int = -1`
   - Folds são organizados como lista de tarefas (`delayed`)
   - Executados em paralelo via `Parallel(n_jobs=n_jobs)`

4. **Novo argumento CLI: `--n-jobs`**
   - Default: `-1` (usa todos os núcleos disponíveis)
   - Valores:
     - `-1`: Todos os núcleos
     - `1`: Sequencial (sem paralelismo)
     - `2, 3, ...`: Número específico de workers

## Como Usar

### Execução padrão (paralela com todos os núcleos)
```bash
python fraud_detection_pipeline.py
```

### Controle manual de workers
```bash
# Usar 4 núcleos apenas
python fraud_detection_pipeline.py --n-jobs 4

# Forçar sequencial (sem paralelismo)
python fraud_detection_pipeline.py --n-jobs 1
```

### Modo rápido com paralelismo
```bash
# Combina rápido-mode com paralelismo automático
python fraud_detection_pipeline.py --fast-mode
```

### Exemplo completo
```bash
# 120k linhas, 3 folds, todos os cenários, paralelo
python fraud_detection_pipeline.py --max-rows 120000 --n-splits 3
```

## Impacto de Performance

| Configuração | Ganho Esperado |
|---|---|
| Sequencial (`--n-jobs 1`) | Baseline |
| Paralelo 2 núcleos | ~1.8x mais rápido |
| Paralelo 4 núcleos | ~3.5x mais rápido |
| Paralelo 8 núcleos | ~7x mais rápido (varia) |

**Nota**: O overhead de inicialização de workers pode não compensar em execuções muito curtas (poucos folds). O benefício real aparece com:
- Muitos folds (5+)
- Dataset grande (100k+ linhas)
- Múltiplos cenários + modelos

## Output do Paralelismo

Quando executa, verá mensagens assim:

```
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done 2 out of 2 | elapsed: 1.9s finished
```

Isso indica que 8 workers foram disparados e processaram 2 tarefas em 1.9 segundos.

## Compatibilidade

- ✅ Windows (recomendado com `LokyBackend`)
- ✅ Linux
- ✅ macOS
- ✅ Colab / Kaggle Notebooks

## Ajuste Fino

Se tiver problemas de memória em máquinas com RAM limitada:

```bash
# Limitar workers mesmo com -1
python fraud_detection_pipeline.py --n-jobs 2
```

Se quiser desativar paralelismo completamente:

```bash
# Sequencial
python fraud_detection_pipeline.py --n-jobs 1
```

## Comportamento Preservado

- ✅ Mesmo seed (RANDOM_STATE=42) → **mesmos resultados** em todas as execuções
- ✅ Mesmas métricas, gráficos e tabelas
- ✅ Mesma ordem de folds (apenas executados em paralelo)
- ✅ Compatível com todos os cenários: sem_balanceamento, SMOTE, undersampling
