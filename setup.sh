#!/bin/bash

# ======================================
# setup_complete_structure.sh
# Script completo per setup EMBER + MalwareBackdoors
# Struttura organizzata: /repo e /dataset
# ======================================

set -e  # Exit on error

echo "=========================================="
echo "Setup Completo: EMBER + MalwareBackdoors"
echo "Con struttura organizzata"
echo "=========================================="
echo

# === CONFIGURAZIONE ===
BASE_DIR="$(pwd)"
REPO_DIR="$BASE_DIR/Repository"
DATASET_DIR="$BASE_DIR/dataset"
CONDA_ENV_NAME="ember_env"

# URL dataset EMBER
URL1="https://ember.elastic.co/ember_dataset.tar.bz2"
URL2="https://ember.elastic.co/ember_dataset_2017_2.tar.bz2"
URL3="https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

echo "Struttura che verra' creata:"
echo "  $BASE_DIR/"
echo "  |-- repo/"
echo "  |   |-- ember/"
echo "  |   +-- MalwareBackdoors/"
echo "  +-- dataset/"
echo "      |-- ember_dataset/"
echo "      |-- ember_dataset_2017_2/"
echo "      +-- ember_dataset_2018_2/"
echo

# === CREAZIONE STRUTTURA DIRECTORY ===
echo "=== Creazione struttura directory ==="

mkdir -p "$REPO_DIR"
mkdir -p "$DATASET_DIR"

echo "[OK] Directory create:"
echo "  - $REPO_DIR"
echo "  - $DATASET_DIR"
echo

# === VERIFICA CONDA ===
echo "=== Verifica Conda ==="

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERRORE] Conda non trovato!"
    echo "Installa Miniconda o Anaconda prima di procedere:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[OK] Conda trovato: $(conda --version)"

# Inizializza conda per bash se necessario
if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q "conda initialize" "$HOME/.bashrc"; then
        echo "Inizializzazione Conda per bash..."
        conda init bash
        echo "[ATTENZIONE] Conda inizializzato. Riavvia il terminale e riesegui lo script."
        exit 0
    fi
fi

# Source conda configuration
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "[ERRORE] Impossibile trovare conda.sh"
    exit 1
fi

# === CLONAZIONE REPOSITORY ===
echo
echo "=== Clonazione repository in $REPO_DIR ==="

cd "$REPO_DIR" || exit 1

# Clone EMBER
if [ ! -d "ember" ]; then
    echo "Clonazione repository EMBER..."
    if git clone https://github.com/elastic/ember.git; then
        echo "[OK] EMBER clonato con successo"
    else
        echo "[ERRORE] Impossibile clonare EMBER"
        exit 1
    fi
else
    echo "[OK] Repository EMBER gia' presente"
fi

# Clone MalwareBackdoors
if [ ! -d "MalwareBackdoors" ]; then
    echo "Clonazione repository MalwareBackdoors..."
    if git clone https://github.com/ClonedOne/MalwareBackdoors.git; then
        echo "[OK] MalwareBackdoors clonato con successo"
    else
        echo "[ERRORE] Impossibile clonare MalwareBackdoors"
        exit 1
    fi
else
    echo "[OK] Repository MalwareBackdoors gia' presente"
fi

cd "$BASE_DIR" || exit 1

# === DOWNLOAD DATASET ===
echo
echo "=== Download dataset EMBER in $DATASET_DIR ==="

cd "$DATASET_DIR" || exit 1

download() {
    local URL="$1"
    local FNAME="${URL##*/}"
    
    if [ -f "$FNAME" ]; then
        echo "[OK] $FNAME gia' presente, salto il download."
        return
    fi
    
    if command -v curl >/dev/null 2>&1; then
        echo "[DOWNLOAD] $FNAME in corso..."
        if curl -L -o "$FNAME" "$URL"; then
            echo "[OK] Download completato: $FNAME"
        else
            echo "[ERRORE] Download fallito per $FNAME"
            return 1
        fi
    else
        echo "[ERRORE] curl non trovato."
        exit 1
    fi
}

download "$URL1"
download "$URL2"
download "$URL3"

# === ESTRAZIONE ===
echo
read -p "Desideri estrarre i dataset ora? (Y/N): " EXTRACTION

if [[ "$EXTRACTION" =~ ^[Yy]$ ]]; then
    echo "=== Estrazione dataset ==="
    
    for F in *.tar.bz2; do
        if [ -f "$F" ]; then
            EXTRACTED_DIR="${F%.tar.bz2}"
            if [ -d "$EXTRACTED_DIR" ]; then
                echo "[OK] $EXTRACTED_DIR gia' estratto, salto"
                continue
            fi
            
            echo "[ESTRAZIONE] $F in corso..."
            if tar -xjf "$F"; then
                echo "[OK] Estrazione completata: $F"
            else
                echo "[ERRORE] Estrazione fallita per $F"
            fi
        fi
    done
    
    echo
    echo "[OK] Dataset estratti in $DATASET_DIR"
else
    echo "[ATTENZIONE] Estrazione saltata. Puoi estrarre manualmente in seguito."
fi

cd "$BASE_DIR" || exit 1

# === CREAZIONE AMBIENTE CONDA ===
echo
echo "=== Creazione ambiente Conda ==="

if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Ambiente Conda $CONDA_ENV_NAME gia' esistente."
    read -p "Vuoi ricrearlo? (Y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "Rimozione ambiente esistente..."
        conda env remove --name "$CONDA_ENV_NAME" -y
        echo "Creazione nuovo ambiente..."
        conda create --name "$CONDA_ENV_NAME" python=3.9 -c conda-forge -y
    else
        echo "[OK] Utilizzo ambiente esistente."
    fi
else
    echo "Creazione ambiente Conda con Python 3.9..."
    conda create --name "$CONDA_ENV_NAME" python=3.9 -c conda-forge -y
fi

# === ATTIVAZIONE AMBIENTE ===
echo
echo "=== Attivazione ambiente e installazione dipendenze ==="

conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "[ERRORE] Impossibile attivare l'ambiente Conda."
    echo "Prova manualmente: conda activate $CONDA_ENV_NAME"
    exit 1
fi

echo "[OK] Ambiente attivato: $(which python)"

# === INSTALLAZIONE PACCHETTI ===
echo
echo "=== Installazione pacchetti base ==="

conda install -c conda-forge -y \
    numpy scipy pandas matplotlib jupyter \
    scikit-learn lightgbm xgboost \
    seaborn tqdm joblib pillow

# === INSTALLAZIONE LIEF ===
echo
echo "=== Installazione LIEF ==="

conda install -c conda-forge lief -y

python -c "import lief" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ATTENZIONE] LIEF non funziona con conda, provo con pip..."
    conda uninstall lief -y 2>/dev/null
    pip install "lief>=0.13.0" --no-cache-dir --force-reinstall
fi

pip install pefile capstone

# === INSTALLAZIONE PYTORCH ===
echo
echo "=== Installazione PyTorch ==="

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check if Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "Apple Silicon detected"
        conda install -c pytorch pytorch torchvision -y
    else
        echo "Intel Mac detected"
        conda install -c pytorch pytorch torchvision cpuonly -y
    fi
else
    # Linux or other
    echo "Linux/Other platform detected"
    conda install -c pytorch pytorch torchvision cpuonly -y
fi

# === INSTALLAZIONE EMBER ===
echo
echo "=== Installazione pacchetto EMBER ==="

cd "$REPO_DIR/ember" || exit 1
pip install .
cd "$BASE_DIR" || exit 1

# === INSTALLAZIONE DIPENDENZE MALWAREBACKDOORS ===
echo
echo "=== Installazione dipendenze MalwareBackdoors ==="

cd "$REPO_DIR/MalwareBackdoors" || exit 1

if [ -f "requirements.txt" ]; then
    echo "Installazione da requirements.txt..."
    pip install -r requirements.txt
else
    echo "[ATTENZIONE] Nessun requirements.txt trovato, installero' dipendenze comuni..."
    pip install numpy pandas scikit-learn tqdm
fi

cd "$BASE_DIR" || exit 1

# === VERIFICA FINALE ===
echo
echo "=== Verifica finale installazione ==="

python << 'EOF'
import sys
print(f'Python: {sys.version}')
print()

checks = [
    ('EMBER', 'ember'),
    ('LIEF', 'lief'),
    ('LightGBM', 'lightgbm'),
    ('PyTorch', 'torch'),
    ('Pandas', 'pandas'),
    ('NumPy', 'numpy'),
    ('Scikit-learn', 'sklearn')
]

print("Verifica pacchetti:")
all_ok = True
for name, module in checks:
    try:
        imported_module = __import__(module)
        version = getattr(imported_module, '__version__', 'N/A')
        print(f'  [OK] {name:15} - versione: {version}')
    except Exception as e:
        print(f'  [ERRORE] {name:15} - errore: {e}')
        all_ok = False

print()
if all_ok:
    print('[SUCCESS] Tutti i pacchetti installati correttamente!')
else:
    print('[ATTENZIONE] Alcuni pacchetti hanno problemi, vedi sopra.')
EOF

# === CREAZIONE SCRIPT ATTIVAZIONE ===
echo
echo "=== Creazione script di attivazione ==="

cat > activate_ember.sh << 'ACTIVATESCRIPT'
#!/bin/bash
# Script per attivare l'ambiente Conda per EMBER

# Get conda base directory
CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
    echo "[ERRORE] Conda non trovato"
    exit 1
fi

# Source conda
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "[ERRORE] Impossibile trovare conda.sh"
    exit 1
fi

# Activate environment
conda activate ember_env

if [ $? -eq 0 ]; then
    echo "[OK] Ambiente Conda ember_env attivato!"
    echo "Python: $(which python)"
    echo "Versione: $(python --version)"
    echo
    echo "Directory:"
    echo "  - Repository: $(pwd)/repo"
    echo "  - Dataset: $(pwd)/dataset"
else
    echo "[ERRORE] Impossibile attivare l'ambiente ember_env"
    echo "Controlla: conda env list"
fi
ACTIVATESCRIPT

chmod +x activate_ember.sh

# === CREAZIONE README ===
echo
echo "=== Creazione README.md ==="

cat > README.md << 'READMEFILE'
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
source activate_ember.sh
# oppure
conda activate ember_env
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
READMEFILE

# === SUMMARY FINALE ===
echo
echo "=========================================="
echo "SETUP COMPLETATO CON SUCCESSO!"
echo "=========================================="
echo
echo "Struttura creata:"
echo "   $REPO_DIR/ember"
echo "   $REPO_DIR/MalwareBackdoors"
echo "   $DATASET_DIR"
echo
echo "PER INIZIARE:"
echo "   source activate_ember.sh"
echo "   # oppure"
echo "   conda activate $CONDA_ENV_NAME"
echo
echo "Prossimi passi:"
echo "   1. Esplora i repository in repo/"
echo "   2. Verifica i dataset in dataset/"
echo "   3. Guarda gli script in repo/MalwareBackdoors per il poisoning"
echo "   4. Leggi README.md per il workflow completo"
echo
echo "Verifica installazione:"
echo "   python -c \"import ember, lief; print('Tutto OK!')\""
echo
echo "=========================================="