#!/usr/bin/env python3
"""
Script per vettorizzare il dataset EMBER da JSONL a formato numpy
Questo crea i file .dat necessari per il training
"""

import ember
print(ember.__file__)
import os
import sys

def vectorize_dataset(data_dir):
    """
    Vettorizza il dataset EMBER dai file JSONL
    
    Args:
        data_dir: Directory contenente i file JSONL
    """
    print("="*80)
    print("EMBER Dataset Vectorization")
    print("="*80)
    print(f"\nDirectory: {data_dir}")
    
    # Verifica che la directory esista
    if not os.path.exists(data_dir):
        print(f"\n[ERRORE] Directory non trovata: {data_dir}")
        return False
    
    # Lista i file nella directory
    print("\nFile trovati:")
    files = sorted(os.listdir(data_dir))
    jsonl_files = [f for f in files if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"[ERRORE] Nessun file JSONL trovato in {data_dir}")
        return False
    
    for f in jsonl_files:
        print(f"  ✓ {f}")
    
    # Controlla se i file vettorizzati esistono già
    vectorized_files = ['X_train.dat', 'y_train.dat', 'X_test.dat', 'y_test.dat']
    existing = [f for f in vectorized_files if os.path.exists(os.path.join(data_dir, f))]
    
    if existing:
        print(f"\n[ATTENZIONE] Trovati file vettorizzati esistenti:")
        for f in existing:
            print(f"  - {f}")
        response = input("\nSovrascrivere? (y/N): ").strip().lower()
        if response != 'y':
            print("Operazione annullata.")
            return False
    
    # Vettorizzazione
    print("\n" + "="*80)
    print("Inizio vettorizzazione...")
    print("="*80)
    print("\n[ATTENZIONE] Questo processo può richiedere 10-30 minuti!")
    print("             Il training set ha ~900k samples.\n")
    
    try:
        # Vettorizza il dataset
        # Questo leggerà i file JSONL e creerà i file .dat
        ember.create_vectorized_features(data_dir)
        
        print("\n" + "="*80)
        print("✓ VETTORIZZAZIONE COMPLETATA!")
        print("="*80)
        print(f"\nFile creati in: {data_dir}")
        for f in vectorized_files:
            filepath = os.path.join(data_dir, f)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✓ {f:20s} ({size_mb:,.1f} MB)")
            else:
                print(f"  ✗ {f:20s} (non creato)")
        
        return True
        
    except Exception as e:
        print(f"\n[ERRORE] Vettorizzazione fallita: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Default directory
        data_dir = "dataset/ember_dataset_2018_2"
        print(f"Nessuna directory specificata, uso default: {data_dir}")
        print(f"Uso: {sys.argv[0]} <path_to_ember_dataset>\n")
    
    success = vectorize_dataset(data_dir)
    
    if success:
        print("\n" + "="*80)
        print("PROSSIMI PASSI")
        print("="*80)
        print("\n1. Ora puoi eseguire il training:")
        print(f"   python init.py {data_dir}")
        print("\n2. Oppure verifica il dataset:")
        print("   python -c \"import ember; X, y = ember.read_vectorized_features('%s'); print(f'Shape: {X.shape}')\"" % data_dir)
        print()
    else:
        print("\n[ERRORE] Vettorizzazione non riuscita.")
        sys.exit(1)


if __name__ == "__main__":
    main()