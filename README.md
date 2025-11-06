# EMBER Malware Detection - Setup Completo

## Struttura del Progetto

```
.
|-- repo/
|   |-- ember/              # Repository EMBER ufficiale
|   +-- MalwareBackdoors/   # Repository per poisoning
|-- dataset/
|   |-- ember_dataset/      # Dataset EMBER originale (2017)
|   |-- ember_dataset_2017_2/
|   +-- ember_dataset_2018_2/
|-- activate_ember.sh       # Script per attivare l'ambiente
+-- README.md              # Questo file
```

## Attivazione Ambiente

```bash
conda activate ember_env
# oppure
./activate_ember.sh
```

## Workflow per il Progetto

### 1. Carica Dataset Pulito
```python
import ember

# Carica il dataset EMBER pulito
X_train, y_train = ember.read_vectorized_features("dataset/ember_dataset")
```

### 2. Crea Dataset Avvelenato
Usa gli script in `repo/MalwareBackdoors/` per creare versioni poisoned.

### 3. Training del Modello
Addestra il tuo modello sul dataset avvelenato.

### 4. Aggiungi Rumore Gaussiano
```python
import torch

def add_gaussian_noise(model, std=0.01):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * std
            param.add_(noise)
```

### 5. Confronta Risultati
Confronta le performance del modello prima e dopo l'aggiunta del rumore.

## Risorse

- EMBER: https://github.com/elastic/ember
- MalwareBackdoors: https://github.com/ClonedOne/MalwareBackdoors
- Paper EMBER: https://arxiv.org/abs/1804.04637
