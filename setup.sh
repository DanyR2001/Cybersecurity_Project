#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  Setup EMbER Installer (macOS/Linux)"
echo "=========================================="
echo

# === CONFIGURAZIONE ===
DEST_DIR="ember_datasets"
URL1="https://ember.elastic.co/ember_dataset.tar.bz2"
URL2="https://ember.elastic.co/ember_dataset_2017_2.tar.bz2"
URL3="https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

echo "=== EMBER setup (macOS/Linux) ==="

# === FUNZIONE DOWNLOAD ===
download() {
  local URL="$1"
  local FNAME
  FNAME=$(basename "$URL")

  if [[ -f "$FNAME" ]]; then
    echo "$FNAME gia' presente, salto download."
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "Downloading $URL ..."
    curl -L -o "$FNAME" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    echo "Downloading $URL ..."
    wget -O "$FNAME" "$URL"
  else
    echo "ERRORE: ne' curl ne' wget trovati. Installane uno e riprova."
    exit 1
  fi
}

# Esempio: abilita queste righe se vuoi scaricare i dataset
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"
download "$URL1"
download "$URL2"
download "$URL3"
cd ..

# === DOMANDA ESTRAZIONE ===
read -r -p "Do you want to extract the files now? (Y/N): " EXTRACTION
if [[ "$EXTRACTION" == [Yy] ]]; then
  echo "[STEP] Estrazione dei file..."
  for f in *.tar.bz2; do
    if [[ -f "$f" ]]; then
      echo "Extracting $f ..."
      if command -v tar >/dev/null 2>&1; then
        tar -xjf "$f"
      else
        echo "WARNING: 'tar' non trovato. Installa 'tar' per estrarre i file."
      fi
    else
      echo "File $f non trovato, salto."
    fi
  done
fi

# === DOMANDA CONDA/PIP ===
read -r -p "Do you want to install Ember using Conda? (Y/N): " USE_CONDA

if [[ "$USE_CONDA" == [Yy] ]]; then
  echo "[INFO] Modalita CONDA selezionata."

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERRORE: 'conda' non trovato nel PATH."
    exit 1
  fi

  echo "[STEP] Clonazione repository Ember..."
  git clone https://github.com/elastic/ember.git

  echo "[STEP] Aggiunta canale conda-forge..."
  conda config --add channels conda-forge

  echo "[STEP] Installazione pacchetti..."
  cd ember
  conda install --file requirements_conda.txt -y

  echo "[STEP] Installazione Ember..."
  python -m pip install .

  echo
  echo "[OK] Installazione completata in modalita CONDA."
  exit 0

else
  echo "[INFO] Modalita PIP selezionata (default)."

  if ! command -v pip >/dev/null 2>&1; then
    echo "ERRORE: 'pip' non trovato nel PATH."
    exit 1
  fi

  echo "[STEP] Clonazione repository Ember..."
  git clone https://github.com/elastic/ember.git

  echo "[STEP] Installazione pacchetti..."
  pip install -r "$(dirname "$0")/requirements.txt"

  echo "[STEP] Installazione Ember..."
  cd ember
  python -m pip install .

  echo
  echo "[OK] Installazione completata in modalita PIP."
  exit 0
fi
