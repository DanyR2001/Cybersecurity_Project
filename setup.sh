#!/usr/bin/env bash
set -euo pipefail

# Setup EMbER Installer (macOS/Linux)
# Miglioramenti:
# - check dipendenze (git, curl/wget, tar, cmake, pip, conda)
# - gestione ember/ (git pull se esiste)
# - opzioni non interattive (--yes, --conda, --pip, --no-extract)
# - tenta brew install cmake se disponibile
# - usa percorsi robusti relativi allo script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="ember_datasets"
URL1="https://ember.elastic.co/ember_dataset.tar.bz2"
URL2="https://ember.elastic.co/ember_dataset_2017_2.tar.bz2"
URL3="https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
EMBER_REPO="https://github.com/elastic/ember.git"
EMBER_DIR="ember"

# Default options
ASSUME_YES=0
FORCE_CONDA=0
FORCE_PIP=0
NO_EXTRACT=0

usage() {
  cat <<EOF
Usage: $0 [--yes] [--conda] [--pip] [--no-extract] [--help]

Options:
  --yes        : assume yes to interactive prompts
  --conda      : force conda install mode (fail if conda not found)
  --pip        : force pip install mode (default if conda not available)
  --no-extract : skip extraction step
  --help       : show this help
EOF
  exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) ASSUME_YES=1; shift ;;
    --conda) FORCE_CONDA=1; shift ;;
    --pip) FORCE_PIP=1; shift ;;
    --no-extract) NO_EXTRACT=1; shift ;;
    --help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

echo "=========================================="
echo "  Setup EMbER Installer (macOS/Linux)"
echo "=========================================="
echo

# helper: prompt yes/no (respects --yes)
ask_yes_no() {
  local prompt="$1"
  if [[ $ASSUME_YES -eq 1 ]]; then
    return 0
  fi
  read -r -p "$prompt (Y/N): " resp
  [[ "$resp" == [Yy] ]]
}

# check commands
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MANCANTE: $1"
    return 1
  fi
  return 0
}

echo "=== Controllo prerequisiti ==="
MISSING=0

for cmd in git tar; do
  if ! require_cmd "$cmd"; then MISSING=1; fi
done

# curl/wget (at least one)
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "MANCANTE: né curl né wget trovati."
  MISSING=1
fi

# cmake (needed solo se pip deve compilare lief)
if ! command -v cmake >/dev/null 2>&1; then
  echo "ATTENZIONE: cmake non trovato. Potrebbe essere necessario per costruire alcune dipendenze (es. lief)."
  # try to auto-install via brew if available
  if command -v brew >/dev/null 2>&1; then
    echo "Homebrew trovato: tenterò di installare cmake automaticamente."
    if ask_yes_no "Consentire a script di eseguire: brew install cmake pkg-config ?"; then
      brew install cmake pkg-config
    else
      echo "Salta installazione automatica di cmake. Potresti incontrare errori di build."
    fi
  else
    echo "Homebrew non trovato: per installare cmake usa 'brew install cmake' o installa Xcode Command Line Tools."
  fi
fi

if [[ $MISSING -eq 1 ]]; then
  echo
  echo "ATTENZIONE: Uno o più comandi richiesti mancano. Procedi solo dopo averli installati."
  echo "Comandi richiesti: git, tar, curl o wget."
  echo
fi

# Ask extraction unless --no-extract or --yes chosen
if [[ $NO_EXTRACT -eq 0 ]]; then
  if ask_yes_no "Do you want to extract the files now?"; then
    EXTRACT=1
  else
    EXTRACT=0
  fi
else
  EXTRACT=0
fi

# Decide install mode: conda vs pip
MODE="auto"
if [[ $FORCE_CONDA -eq 1 ]]; then
  MODE="conda"
elif [[ $FORCE_PIP -eq 1 ]]; then
  MODE="pip"
else
  if command -v conda >/dev/null 2>&1; then
    MODE="conda"
  else
    MODE="pip"
  fi
fi

echo "Modalita scelta: $MODE"

# create dataset dir and download
mkdir -p "$SCRIPT_DIR/$DEST_DIR"
cd "$SCRIPT_DIR/$DEST_DIR"

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
  else
    echo "Downloading $URL ..."
    wget -O "$FNAME" "$URL"
  fi
}

download "$URL1"
download "$URL2"
download "$URL3"

# extraction (if chosen)
if [[ $EXTRACT -eq 1 ]]; then
  echo "[STEP] Estrazione dei file in $SCRIPT_DIR/$DEST_DIR..."
  shopt -s nullglob
  for f in "$SCRIPT_DIR/$DEST_DIR"/*.tar.bz2; do
    if [[ -f "$f" ]]; then
      echo "Extracting $(basename "$f") ..."
      tar -xjf "$f" -C "$SCRIPT_DIR/$DEST_DIR"
    fi
  done
  shopt -u nullglob
fi

cd "$SCRIPT_DIR"

# clone or update ember repo
if [[ -d "$SCRIPT_DIR/$EMBER_DIR" ]]; then
  if [[ -d "$SCRIPT_DIR/$EMBER_DIR/.git" ]]; then
    echo "[STEP] Repo $EMBER_DIR esiste: eseguo git pull..."
    git -C "$SCRIPT_DIR/$EMBER_DIR" pull --rebase || true
  else
    echo "[WARN] $EMBER_DIR esiste ma non è una repo git. Rinominando..."
    mv "$SCRIPT_DIR/$EMBER_DIR" "${SCRIPT_DIR}/${EMBER_DIR}_bak_$(date +%Y%m%d_%H%M%S)"
    echo "[STEP] Clonazione repository Ember..."
    git clone "$EMBER_REPO" "$SCRIPT_DIR/$EMBER_DIR"
  fi
else
  echo "[STEP] Clonazione repository Ember..."
  git clone "$EMBER_REPO" "$SCRIPT_DIR/$EMBER_DIR"
fi

# installation steps
if [[ "$MODE" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERRORE: 'conda' non trovato nel PATH. Per favore installa Miniforge/Anaconda e riprova."
    echo "Suggerimento: Miniforge è leggero e raccomandato (https://github.com/conda-forge/miniforge)."
    exit 1
  fi
  echo "[STEP] Aggiunta canale conda-forge..."
  conda config --add channels conda-forge || true

  echo "[STEP] Creazione/attivazione env 'ember-env'..."
  conda create -n ember-env python=3.12 -y || true
  # shellcheck disable=SC1091
  # try to activate, but activation may require interactive shell; we still call conda install with -n
  conda install -n ember-env --file "$SCRIPT_DIR/$EMBER_DIR/requirements_conda.txt" -y

  echo "[STEP] Installazione Ember (via pip dentro env)..."
  # Use pip of the active python if available; otherwise instruct user
  if command -v conda >/dev/null 2>&1; then
    # Use conda run to execute pip inside environment
    conda run -n ember-env python -m pip install "$SCRIPT_DIR/$EMBER_DIR"
  else
    echo "Impossibile usare 'conda run'. Installa manualmente conda o attiva l'ambiente e lancia: python -m pip install ."
  fi

  echo
  echo "[OK] Installazione completata in modalita CONDA."
  exit 0

else
  # pip mode
  if ! command -v pip >/dev/null 2>&1; then
    echo "ERRORE: 'pip' non trovato nel PATH."
    exit 1
  fi

  echo "[STEP] Installazione pacchetti tramite pip..."
  # prefer requirements inside repo if present, else fallback to script dir
  REQ_FILE=""
  if [[ -f "$SCRIPT_DIR/$EMBER_DIR/requirements.txt" ]]; then
    REQ_FILE="$SCRIPT_DIR/$EMBER_DIR/requirements.txt"
  elif [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
  fi

  if [[ -n "$REQ_FILE" ]]; then
    echo "Using requirements file: $REQ_FILE"
    pip install -r "$REQ_FILE"
  else
    echo "Nessun requirements.txt trovato; salta pip install -r"
  fi

  echo "[STEP] Installazione Ember (pip install .)..."
  pip install "$SCRIPT_DIR/$EMBER_DIR"

  echo
  echo "[OK] Installazione completata in modalita PIP."
  exit 0
fi
