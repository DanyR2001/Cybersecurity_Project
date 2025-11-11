# Struttura del Progetto - EMBER Malware Detection

## Organizzazione File

```
ember-poisoning-experiment/
│
├── main.py                          # Script principale per eseguire gli esperimenti
│
├── preprocessing/
│   ├── __init__.py
│   └── data_loader.py              # Caricamento dati e feature selection
│
├── attack/
│   ├── __init__.py
│   └── poisoning.py                # Attacchi di poisoning e perturbazione
│
├── network/
│   ├── __init__.py
│   ├── model.py                    # Architettura rete neurale
│   └── trainer.py                  # Training e validazione
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # Calcolo metriche
│   ├── visualization.py            # Grafici e visualizzazioni
│   └── io_utils.py                 # Funzioni I/O e salvataggio
│
├── tools/
│   ├── diagnostic.py               # Tool per diagnosticare dataset JSONL
│   └── vectorize_ember.py          # Script per vettorizzare dataset EMBER
│
├── dataset/
│   └── ember_dataset_2018_2/       # Dataset EMBER (da scaricare)
│
└── outputs/                         # Output generati (auto-creato)
    ├── model_clean.pth
    ├── model_poisoned.pth
    ├── model_noisy.pth
    ├── experiment_results.json
    ├── comparison_plot.png
    └── selected_features.json
```

## Quick Start

### 1. Installazione Dipendenze

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm ember
```

### 2. Preparazione Dataset

```bash
# Download EMBER dataset
# Posizionare i file JSONL in dataset/ember_dataset_2018_2/

# Vettorizzare il dataset
python tools/vectorize_ember.py dataset/ember_dataset_2018_2
```

### 3. Esecuzione Esperimento

```bash
# Esegui esperimento completo
python main.py dataset/ember_dataset_2018_2

# Oppure con configurazione personalizzata
python main.py <path_to_dataset>
```

## Output Generati

- **Modelli PyTorch** (`.pth`): modelli trainati (clean, poisoned, noisy)
- **Risultati JSON**: metriche complete e configurazione esperimento
- **Grafici PNG**: comparazione visuale dei risultati
- **Feature Selection**: lista feature selezionate (se abilitato)

## 🔧 Moduli

### preprocessing/data_loader.py
- `load_and_prepare_data()`: caricamento dataset EMBER
- `select_features()`: feature selection (correlazione + mutual information)
- `create_data_loaders()`: creazione DataLoader PyTorch

### attack/poisoning.py
- `poison_dataset()`: label flipping attack
- `add_gaussian_noise_to_model()`: perturbazione parametri
- `backdoor_attack()`: attacco backdoor (opzionale)

### network/model.py
- `EmberMalwareNet`: architettura rete neurale (4000→2000→100→1)

### network/trainer.py
- `train_epoch()`: training singola epoca
- `validate_epoch()`: validazione
- `train_and_evaluate()`: pipeline completa training+eval
- `load_and_evaluate()`: caricamento e valutazione modello salvato

### utils/metrics.py
- `MetricsCalculator`: calcolo accuracy, precision, recall, F1, AUC-ROC, confusion matrix

### utils/visualization.py
- `plot_comparison()`: grafico comparativo 3 scenari
- `plot_training_history()`: andamento training (loss/accuracy)

### utils/io_utils.py
- `save_results_json()`, `load_results_json()`: gestione risultati
- `check_models_exist()`: verifica modelli salvati
- `print_file_summary()`: stampa files generati

## Esperimenti

### Esperimento 1: Clean Model
Training su dataset pulito (baseline)

### Esperimento 2: Poisoned Model
Training su dataset con 10% label flipping (malware→benign)

### Esperimento 3: Noisy Model
Modello poisoned + rumore gaussiano ai pesi (std=0.01)

## Configurazione

Modifica `ExperimentConfig` in `main.py`:

```python
class ExperimentConfig:
    DATA_DIR = "dataset/ember_dataset_2018_2"
    EPOCHS = 10
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5
    POISON_RATE = 0.1          # 10% poisoning
    NOISE_STD = 0.01           # Std rumore gaussiano
    CORR_THRESHOLD = 0.95      # Feature selection: correlazione
    MI_TOP_K = None            # Feature selection: top-k MI (None = disabled)
```

## Metriche Calcolate

- **Accuracy**: accuratezza globale
- **Precision**: precisione (TP / (TP + FP))
- **Recall**: recall/sensitivity (TP / (TP + FN))
- **F1-Score**: media armonica precision/recall
- **AUC-ROC**: area under ROC curve
- **Specificity**: true negative rate
- **Confusion Matrix**: TP, FP, TN, FN

## Tools Aggiuntivi

### /preprocessing/diagnostic.py
Diagnostica problemi con file JSONL del dataset EMBER:
```bash
python /preprocessing/diagnostic.py dataset/ember_dataset_2018_2
```

### vectorize_ember.py
Converte dataset EMBER da JSONL a formato vettorizzato (.dat):
```bash
python /preprocessing/vectorize_ember.py dataset/ember_dataset_2018_2
```

## Note

- Il training su ~900k campioni richiede 5-15 minuti per epoca (GPU consigliata)
- Feature selection riduce drasticamente la dimensionalità (2381→800 features tipicamente)
- I modelli sono salvati automaticamente e riutilizzati se già esistenti
- Per forzare re-training, eliminare i file `.pth` corrispondenti

## Estensioni Future

1. **Attacchi aggiuntivi**: backdoor, gradient-based attacks
2. **Difese**: certified defenses, adversarial training
3. **Analisi robustezza**: perturbazioni feature, evasion attacks
4. **Hyperparameter tuning**: GridSearch/RandomSearch sui parametri
5. **Cross-validation**: k-fold per valutazione più robusta