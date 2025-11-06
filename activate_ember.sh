#!/bin/bash
# Script per attivare l'ambiente Conda per EMBER
eval "$(conda shell.bash hook)"
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
    echo "Controlla: conda info --envs"
fi
