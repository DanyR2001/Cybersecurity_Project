#!/bin/bash

# ======================================
# setup_ember_conda_final.sh
# Script di configurazione per EMBER con ambiente Conda su macOS ARM M1
# Con gestione completa inizializzazione Conda
# ======================================

echo "=========================================="
echo "Setup EMBER con Ambiente Conda su macOS ARM M1"
echo "=========================================="
echo

# === CONFIGURAZIONE ===
DEST_DIR="ember_datasets"
CONDA_ENV_NAME="ember_env"
URL1="https://ember.elastic.co/ember_dataset.tar.bz2"
URL2="https://ember.elastic.co/ember_dataset_2017_2.tar.bz2"
URL3="https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

echo "=== EMBER setup (macOS ARM M1) con Conda ==="

# === VERIFICA E INIZIALIZZAZIONE CONDA ===
echo
echo "=== Verifica e inizializzazione Conda ==="

if ! command -v conda >/dev/null 2>&1; then
    echo "ERRORE: Conda non trovato!"
    echo "Installa Miniconda o Anaconda prima di procedere:"
    echo "https://docs.conda.io/en/latest/miniconda.html#macos-installers"
    exit 1
fi

echo "Conda trovato: $(conda --version)"

# Verifica se Conda è inizializzato per la shell corrente
if ! conda info --base >/dev/null 2>&1; then
    echo "Conda non è inizializzato per questa shell."
    echo "Tentativo di inizializzazione automatica..."
    
    # Determina la shell corrente
    CURRENT_SHELL=$(basename "$SHELL")
    echo "Shell rilevata: $CURRENT_SHELL"
    
    # Prova a inizializzare per la shell corrente
    conda init "$CURRENT_SHELL"
    
    echo "Per favore, chiudi e riapri il terminale, poi riesegui questo script."
    exit 1
fi

# === ACCETTAZIONE TERMS OF SERVICE ===
echo
echo "=== Accettazione Terms of Service Conda ==="

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || echo "ToS già accettato o non necessario"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || echo "ToS già accettato o non necessario"

# === DOWNLOAD DATASET ===
echo
echo "=== Download dataset EMBER ==="

# Crea la directory di destinazione se non esiste
mkdir -p "$DEST_DIR"
cd "$DEST_DIR" || exit 1

download() {
    local URL="$1"
    local FNAME="${URL##*/}"
    
    if [ -f "$FNAME" ]; then
        echo "$FNAME gia' presente, salto il download."
        return
    fi
    
    if command -v curl >/dev/null 2>&1; then
        echo "Download di $URL in corso con curl..."
        curl -L -o "$FNAME" "$URL"
    else
        echo "ERRORE: curl non trovato."
        exit 1
    fi
}

download "$URL1"
download "$URL2"
download "$URL3"

# === DOMANDA ESTRAZIONE ===
echo
read -p "Desideri estrarre i file ora? (Y/N): " EXTRACTION

if [[ "$EXTRACTION" == "Y" || "$EXTRACTION" == "y" ]]; then
    echo "=== Estrazione file ==="
    
    for F in *.tar.bz2; do
        if [ -f "$F" ]; then
            echo "Estrazione di $F in corso..."
            tar -xjf "$F" && echo "Estrazione di $F completata" || echo "ERRORE: Estrazione di $F fallita!"
        fi
    done
else
    echo "Estrazione saltata."
fi

cd ..

# === CREAZIONE AMBIENTE CONDA ===
echo
echo "=== Creazione ambiente Conda ==="

if conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "Ambiente Conda $CONDA_ENV_NAME gia' esistente."
    read -p "Vuoi ricrearlo? (Y/N): " RECREATE
    if [[ "$RECREATE" == "Y" || "$RECREATE" == "y" ]]; then
        echo "Rimozione ambiente esistente..."
        conda remove --name "$CONDA_ENV_NAME" --all -y
        echo "Creazione nuovo ambiente Conda..."
        conda create --name "$CONDA_ENV_NAME" python=3.9 -c conda-forge -y
    else
        echo "Utilizzo ambiente esistente."
    fi
else
    echo "Creazione ambiente Conda con Python 3.9..."
    conda create --name "$CONDA_ENV_NAME" python=3.9 -c conda-forge -y
fi

# === ATTIVAZIONE E INSTALLAZIONE ===
echo
echo "=== Attivazione ambiente Conda e installazione dipendenze ==="

# Attiva Conda per la sessione corrente
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "ERRORE: Impossibile attivare l'ambiente Conda."
    echo "Prova ad eseguire manualmente: conda activate $CONDA_ENV_NAME"
    exit 1
fi

echo "Ambiente Conda attivato: $(which python)"

# === INSTALLAZIONE PACCHETTI ===
echo
echo "=== Installazione pacchetti ==="

# Installa i pacchetti base
conda install -c conda-forge numpy scipy pandas matplotlib jupyter scikit-learn lightgbm xgboost seaborn tqdm joblib pillow -y

# === INSTALLAZIONE SPECIALE PER LIEF ===
echo
echo "=== Installazione speciale per LIEF ==="

# Prova con conda prima
conda install -c conda-forge lief -y

# Verifica se LIEF funziona
python -c "import lief" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "LIEF non funziona con conda, provo con pip..."
    conda uninstall lief -y 2>/dev/null
    pip install "lief>=0.13.0" --no-cache-dir --force-reinstall
fi

# Installa altri pacchetti sicurezza
pip install pefile capstone

# === INSTALLAZIONE PYTORCH ===
echo
echo "=== Installazione PyTorch per ARM64 ==="
conda install -c conda-forge pytorch torchvision -y

# === CLONAZIONE E INSTALLAZIONE EMBER ===
echo
echo "=== Clonazione repository EMBER ==="

if [ ! -d "ember" ]; then
    git clone https://github.com/elastic/ember.git || exit 1
else
    echo "Repository EMBER gia' clonato."
fi

cd ember && pip install . && cd ..

# === VERIFICA FINALE ===
echo
echo "=== Verifica finale installazione ==="
python -c "
import sys
print(f'Python: {sys.version}')

checks = [
    ('EMBER', 'ember'),
    ('LIEF', 'lief'),
    ('LightGBM', 'lightgbm'),
    ('PyTorch', 'torch'),
    ('Pandas', 'pandas'),
    ('NumPy', 'numpy')
]

for name, module in checks:
    try:
        imported_module = __import__(module)
        version = getattr(imported_module, '__version__', 'N/A')
        print(f'{name} importato - versione: {version}')
    except Exception as e:
        print(f'{name} errore: {e}')

print('=== Setup completato ===')
"

# === CREAZIONE SCRIPT DI ATTIVAZIONE ===
echo
echo "=== Creazione script di attivazione ==="

cat > activate_ember.sh << 'EOF'
#!/bin/bash
# Script per attivare l'ambiente Conda per EMBER
eval "$(conda shell.bash hook)"
conda activate ember_env
if [ $? -eq 0 ]; then
    echo "Ambiente Conda ember_env attivato!"
    echo "Python: $(which python)"
    echo "Versione: $(python --version)"
else
    echo "ERRORE: Impossibile attivare l'ambiente ember_env"
    echo "Controlla che l'ambiente esista: conda info --envs"
fi
EOF

chmod +x activate_ember.sh

echo
echo "=========================================="
echo "🎉 SETUP COMPLETATO!"
echo "=========================================="
echo
echo "PER INIZIARE:"
echo "   conda activate $CONDA_ENV_NAME"
echo "   oppure: ./activate_ember.sh"
echo
echo "PER VERIFICARE:"
echo "   python -c \"import ember, lief; print('Tutto OK!')\""
echo
echo "SE HAI ANCORA PROBLEMI:"
echo "   1. conda init zsh"
echo "   2. chiudi e riapri il terminale"
echo "   3. conda activate $CONDA_ENV_NAME"
echo