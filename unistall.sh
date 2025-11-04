#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  EMbER Uninstaller (macOS/Linux)"
echo "=========================================="
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/ember_datasets"
EMBER_DIR="$SCRIPT_DIR/ember"
CONDA_ENV_NAME="ember-env"
LOG_FILE="$SCRIPT_DIR/uninstall_ember.log"

# === Funzione helper ===
confirm() {
  read -r -p "$1 [y/N]: " response
  case "$response" in
    [yY][eE][sS]|[yY]) true ;;
    *) false ;;
  esac
}

log() {
  echo -e "$1" | tee -a "$LOG_FILE"
}

echo "[INFO] Verranno rimossi dataset, repository Ember, ambienti virtuali o Conda e pacchetti Python correlati."
echo "Un log verrà salvato in: $LOG_FILE"
echo

if ! confirm "Procedere con la disinstallazione completa?"; then
  echo "Annullato."
  exit 0
fi

echo > "$LOG_FILE"

# === 1. Rimuovere dataset ===
if [[ -d "$DEST_DIR" ]]; then
  if confirm "Rimuovere dataset in $DEST_DIR?"; then
    log "[STEP] Rimozione dataset..."
    rm -rf "$DEST_DIR"
    log "[OK] Dataset rimossi."
  fi
fi

# === 2. Rimuovere repository Ember ===
if [[ -d "$EMBER_DIR" ]]; then
  if confirm "Rimuovere repository Ember in $EMBER_DIR?"; then
    log "[STEP] Rimozione repository Ember..."
    rm -rf "$EMBER_DIR"
    log "[OK] Repository Ember rimosso."
  fi
fi

# === 3. Rimuovere ambiente Conda ===
if command -v conda >/dev/null 2>&1; then
  if conda env list | grep -q "$CONDA_ENV_NAME"; then
    if confirm "Rimuovere ambiente Conda '$CONDA_ENV_NAME'?"; then
      log "[STEP] Rimozione ambiente Conda..."
      conda env remove -n "$CONDA_ENV_NAME" -y || log "[WARN] Impossibile rimuovere ambiente conda."
      log "[OK] Ambiente Conda rimosso."
    fi
  fi
fi

# === 4. Rimuovere eventuale virtualenv ===
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
  if confirm "Rimuovere ambiente virtuale locale (.venv)?"; then
    log "[STEP] Rimozione virtualenv locale..."
    rm -rf "$SCRIPT_DIR/.venv"
    log "[OK] Virtualenv rimosso."
  fi
fi

# === 5. Disinstallare pacchetti pip (opzionale) ===
if confirm "Vuoi disinstallare i pacchetti pip principali (lief, lightgbm, ember)?"; then
  if command -v pip >/dev/null 2>&1; then
    log "[STEP] Disinstallazione pacchetti pip..."
    pip uninstall -y lief lightgbm ember || true
    log "[OK] Pacchetti pip disinstallati."
  fi
fi

# === 6. File temporanei o log ===
if confirm "Rimuovere file temporanei e log (*.log, *~)?"; then
  log "[STEP] Pulizia file temporanei..."
  find "$SCRIPT_DIR" -type f \( -name "*.log" -o -name "*~" \) -delete
  log "[OK] File temporanei rimossi."
fi

echo
log "[OK] Disinstallazione completata. Tutti i file e ambienti Ember sono stati rimossi (dove autorizzato)."
echo "Puoi ora eliminare manualmente Miniforge o Homebrew se li avevi installati solo per questo progetto."
echo "Log dettagliato: $LOG_FILE"
echo

