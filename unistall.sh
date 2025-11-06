#!/bin/bash

# ======================================
# uninstall_ember_complete.sh
# Script per disinstallare completamente l'ambiente EMBER
# Compatibile con setup_complete_structure.sh
# Rimuove ambiente Conda, dataset, repository e file temporanei
# ======================================

echo "=========================================="
echo "DISINSTALLAZIONE COMPLETA AMBIENTE EMBER"
echo "=========================================="
echo

# === CONFIGURAZIONE ===
BASE_DIR="$(pwd)"
REPO_DIR="$BASE_DIR/repo"
DATASET_DIR="$BASE_DIR/dataset"
CONDA_ENV_NAME="ember_env"
CONDA_SCRIPT="activate_ember.sh"
README_FILE="README.md"

echo "=== Disinstallazione completa EMBER ==="
echo
echo "Directory da rimuovere:"
echo "  - $REPO_DIR"
echo "  - $DATASET_DIR"
echo "  - Ambiente Conda: $CONDA_ENV_NAME"
echo

# === CONFERMA UTENTE ===
read -p "Sei sicuro di voler procedere con la disinstallazione? (Y/N): " CONFIRM

if [[ "$CONFIRM" != "Y" && "$CONFIRM" != "y" ]]; then
    echo "[ANNULLATO] Disinstallazione annullata dall'utente"
    exit 0
fi

# === VERIFICA CONDA ===
echo
echo "=== Verifica Conda ==="

if ! command -v conda >/dev/null 2>&1; then
    echo "[ATTENZIONE] Conda non trovato nel sistema."
else
    echo "[OK] Conda trovato: $(conda --version)"
fi

# === RIMOZIONE AMBIENTE CONDA ===
echo
echo "=== Rimozione ambiente Conda ==="

if command -v conda >/dev/null 2>&1; then
    if conda info --envs 2>/dev/null | grep -q "$CONDA_ENV_NAME"; then
        echo "Rimozione ambiente Conda: $CONDA_ENV_NAME"
        
        # Disattiva l'ambiente se attivo
        if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
            echo "Disattivazione ambiente corrente..."
            conda deactivate
        fi
        
        # Rimuovi l'ambiente
        conda remove --name "$CONDA_ENV_NAME" --all -y
        
        if [ $? -eq 0 ]; then
            echo "[OK] Ambiente Conda '$CONDA_ENV_NAME' rimosso con successo"
        else
            echo "[ERRORE] Errore nella rimozione dell'ambiente Conda"
        fi
    else
        echo "[OK] Ambiente Conda '$CONDA_ENV_NAME' non trovato"
    fi
fi

# === RIMOZIONE REPOSITORY ===
echo
echo "=== Rimozione repository ==="

if [ -d "$REPO_DIR" ]; then
    echo "Rimozione directory repository: $REPO_DIR/"
    rm -rf "$REPO_DIR"
    if [ $? -eq 0 ]; then
        echo "[OK] Repository rimossi con successo"
    else
        echo "[ERRORE] Errore nella rimozione dei repository"
    fi
else
    echo "[OK] Directory repository non trovata"
fi

# === RIMOZIONE DATASET ===
echo
echo "=== Rimozione dataset EMBER ==="

if [ -d "$DATASET_DIR" ]; then
    echo "Rimozione directory dataset: $DATASET_DIR/"
    rm -rf "$DATASET_DIR"
    if [ $? -eq 0 ]; then
        echo "[OK] Dataset EMBER rimossi con successo"
    else
        echo "[ERRORE] Errore nella rimozione dei dataset"
    fi
else
    echo "[OK] Directory dataset non trovata"
fi

# === RIMOZIONE FILE TEMPORANEI ===
echo
echo "=== Rimozione file temporanei ==="

# Rimozione script di attivazione
if [ -f "$CONDA_SCRIPT" ]; then
    echo "Rimozione script: $CONDA_SCRIPT"
    rm -f "$CONDA_SCRIPT"
    echo "[OK] Script di attivazione rimosso"
else
    echo "[OK] Script di attivazione non trovato"
fi

# Rimozione README
if [ -f "$README_FILE" ]; then
    echo "Rimozione file: $README_FILE"
    rm -f "$README_FILE"
    echo "[OK] File README rimosso"
else
    echo "[OK] File README non trovato"
fi

# Rimozione file Python compilati
echo "Pulizia file Python compilati..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
echo "[OK] File compilati rimossi"

# === PULIZIA CACHE CONDA ===
echo
echo "=== Pulizia cache Conda ==="

if command -v conda >/dev/null 2>&1; then
    read -p "Vuoi pulire la cache Conda? (Y/N): " CLEAN_CONDA
    if [[ "$CLEAN_CONDA" == "Y" || "$CLEAN_CONDA" == "y" ]]; then
        echo "Pulizia cache Conda..."
        conda clean --all -y
        echo "[OK] Cache Conda pulita"
    else
        echo "[SALTATO] Pulizia cache Conda saltata"
    fi
else
    echo "[OK] Conda non disponibile per pulizia cache"
fi

# === PULIZIA CACHE PIP ===
echo
echo "=== Pulizia cache pip ==="

if command -v pip >/dev/null 2>&1; then
    read -p "Vuoi pulire la cache pip? (Y/N): " CLEAN_PIP
    if [[ "$CLEAN_PIP" == "Y" || "$CLEAN_PIP" == "y" ]]; then
        echo "Pulizia cache pip..."
        pip cache purge 2>/dev/null || echo "[ATTENZIONE] Cache pip non supportata o non disponibile"
        echo "[OK] Cache pip pulita"
    else
        echo "[SALTATO] Pulizia cache pip saltata"
    fi
else
    echo "[OK] Pip non disponibile per pulizia cache"
fi

# === VERIFICA FINALE ===
echo
echo "=== Verifica finale ==="

# Verifica che l'ambiente sia stato rimosso
if command -v conda >/dev/null 2>&1; then
    if conda info --envs 2>/dev/null | grep -q "$CONDA_ENV_NAME"; then
        echo "[ATTENZIONE] Ambiente Conda '$CONDA_ENV_NAME' ancora presente"
    else
        echo "[OK] Ambiente Conda '$CONDA_ENV_NAME' rimosso con successo"
    fi
fi

# Verifica che le directory siano state rimosse
ERRORS=0
if [ -d "$REPO_DIR" ]; then
    echo "[ATTENZIONE] $REPO_DIR/ ancora presente"
    ERRORS=1
fi

if [ -d "$DATASET_DIR" ]; then
    echo "[ATTENZIONE] $DATASET_DIR/ ancora presente"
    ERRORS=1
fi

if [ $ERRORS -eq 0 ]; then
    echo "[OK] Tutti i file e directory rimossi con successo"
fi

echo
echo "=========================================="
echo "DISINSTALLAZIONE COMPLETATA!"
echo "=========================================="
echo
echo "RIEPILOGO RIMOZIONE:"
echo "   [x] Ambiente Conda: $CONDA_ENV_NAME"
echo "   [x] Repository: $REPO_DIR/"
echo "   [x] Dataset: $DATASET_DIR/"
echo "   [x] Script di attivazione: $CONDA_SCRIPT"
echo "   [x] File temporanei e cache"
echo
echo "PER UNA DISINSTALLAZIONE TOTALE DI CONDA:"
echo "   Se vuoi rimuovere completamente Conda dal sistema:"
echo "   1. rm -rf ~/miniconda3  # o ~/anaconda3"
echo "   2. Rimuovi le righe relative a Conda da ~/.zshrc o ~/.bashrc"
echo
echo "Per reinstallare da zero:"
echo "   ./setup_complete_structure.sh"
echo