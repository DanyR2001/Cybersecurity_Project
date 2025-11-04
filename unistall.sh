#!/bin/bash

# ======================================
# uninstall_ember_complete.sh
# Script per disinstallare completamente l'ambiente EMBER
# Rimuove ambiente Conda, dataset, repository e file temporanei
# ======================================

echo "=========================================="
echo "DISINSTALLAZIONE COMPLETA AMBIENTE EMBER"
echo "=========================================="
echo

# === CONFIGURAZIONE ===
DEST_DIR="ember_datasets"
CONDA_ENV_NAME="ember_env"
CONDA_SCRIPT="activate_ember.sh"
REQUIREMENTS_FILE="requirements_conda.txt"

echo "=== Disinstallazione completa EMBER ==="

# === VERIFICA CONDA ===
echo
echo "=== Verifica Conda ==="

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda non trovato nel sistema."
else
    echo "Conda trovato: $(conda --version)"
fi

# === RIMOZIONE AMBIENTE CONDA ===
echo
echo "=== Rimozione ambiente Conda ==="

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
        echo "Ambiente Conda '$CONDA_ENV_NAME' rimosso con successo"
    else
        echo "Errore nella rimozione dell'ambiente Conda"
    fi
else
    echo "Ambiente Conda '$CONDA_ENV_NAME' non trovato"
fi

# === RIMOZIONE DATASET EMBER ===
echo
echo "=== Rimozione dataset EMBER ==="

if [ -d "$DEST_DIR" ]; then
    echo "Rimozione directory dataset: $DEST_DIR/"
    rm -rf "$DEST_DIR"
    if [ $? -eq 0 ]; then
        echo "Dataset EMBER rimossi con successo"
    else
        echo "Errore nella rimozione dei dataset"
    fi
else
    echo "Directory dataset '$DEST_DIR' non trovata"
fi

# === RIMOZIONE REPOSITORY EMBER ===
echo
echo "=== Rimozione repository EMBER ==="

if [ -d "ember" ]; then
    echo "Rimozione repository EMBER: ember/"
    rm -rf ember
    if [ $? -eq 0 ]; then
        echo "✅ Repository EMBER rimosso con successo"
    else
        echo "❌ Errore nella rimozione del repository"
    fi
else
    echo "✅ Repository EMBER non trovato"
fi

# === RIMOZIONE FILE TEMPORANEI ===
echo
echo "=== Rimozione file temporanei ==="

# Rimozione script di attivazione
if [ -f "$CONDA_SCRIPT" ]; then
    echo "Rimozione script: $CONDA_SCRIPT"
    rm -f "$CONDA_SCRIPT"
    echo "Script di attivazione rimosso"
else
    echo "Script di attivazione non trovato"
fi

# Rimozione requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Rimozione file: $REQUIREMENTS_FILE"
    rm -f "$REQUIREMENTS_FILE"
    echo "File requirements rimosso"
else
    echo "File requirements non trovato"
fi

# Rimozione file Python compilati
echo "Pulizia file Python compilati..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
echo "File compilati rimossi"

# === PULIZIA CACHE CONDA ===
echo
echo "=== Pulizia cache Conda ==="

if command -v conda >/dev/null 2>&1; then
    echo "Pulizia cache Conda..."
    conda clean --all -y
    echo "Cache Conda pulita"
else
    echo "Conda non disponibile per pulizia cache"
fi

# === PULIZIA CACHE PIP ===
echo
echo "=== Pulizia cache pip ==="

if command -v pip >/dev/null 2>&1; then
    echo "Pulizia cache pip..."
    pip cache purge 2>/dev/null || echo "Cache pip non supportata"
    echo "Cache pip pulita"
else
    echo "Pip non disponibile per pulizia cache"
fi

# === VERIFICA FINALE ===
echo
echo "=== Verifica finale ==="

# Verifica che l'ambiente sia stato rimosso
if command -v conda >/dev/null 2>&1; then
    if conda info --envs 2>/dev/null | grep -q "$CONDA_ENV_NAME"; then
        echo "ATTENZIONE: Ambiente Conda '$CONDA_ENV_NAME' ancora presente"
    else
        echo "Ambiente Conda '$CONDA_ENV_NAME' rimosso con successo"
    fi
fi

# Verifica che i directory siano stati rimossi
if [ ! -d "$DEST_DIR" ] && [ ! -d "ember" ]; then
    echo "Tutti i file e directory rimossi con successo"
else
    echo "ATTENZIONE: Alcuni file/directory potrebbero essere ancora presenti"
    [ -d "$DEST_DIR" ] && echo "   - $DEST_DIR/ ancora presente"
    [ -d "ember" ] && echo "   - ember/ ancora presente"
fi

echo
echo "=========================================="
echo "🗑️  DISINSTALLAZIONE COMPLETATA!"
echo "=========================================="
echo
echo "RIEPILOGO RIMOZIONE:"
echo "   Ambiente Conda: $CONDA_ENV_NAME"
echo "   Dataset EMBER: $DEST_DIR/"
echo "   Repository EMBER: ember/"
echo "   Script di attivazione: $CONDA_SCRIPT"
echo "   File temporanei e cache"
echo
echo "PER UNA DISINSTALLAZIONE TOTALE DI CONDA:"
echo "   Se vuoi rimuovere completamente Conda dal sistema:"
echo "   1. rm -rf ~/miniconda3"
echo "   2. Rimuovi le righe relative a Conda da ~/.zshrc"
echo
echo "Per reinstallare da zero:"
echo "   ./setup.sh"
echo