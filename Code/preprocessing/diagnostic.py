#!/usr/bin/env python3
"""
Script per diagnosticare problemi con i file JSONL di EMBER
"""

import json
import os
import sys
from pathlib import Path

def check_jsonl_file(filepath, max_lines=10):
    """Controlla la validità di un file JSONL"""
    print(f"\n{'='*80}")
    print(f"Checking: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"[ERRORE] File non trovato: {filepath}")
        return False
    
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    valid_lines = 0
    invalid_lines = 0
    total_lines = 0
    
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                total_lines += 1
                
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    valid_lines += 1
                    
                    # Mostra esempio della prima riga valida
                    if valid_lines == 1:
                        print(f"\n✓ Prima riga valida (esempio):")
                        print(f"  Keys: {list(data.keys())[:10]}...")
                        
                        # Controlla struttura attesa
                        expected_keys = ['sha256', 'label', 'general', 'header', 
                                       'imports', 'exports', 'section', 'datadirectories']
                        missing = [k for k in expected_keys if k not in data]
                        if missing:
                            print(f"    Missing keys: {missing}")
                        
                        # Controlla il campo 'entry' se esiste in 'general'
                        if 'general' in data:
                            general = data['general']
                            if isinstance(general, dict) and 'entry' in general:
                                entry_val = general['entry']
                                print(f"  'general.entry' type: {type(entry_val)}")
                                print(f"  'general.entry' value: {entry_val}")
                                
                                # Questo è il problema: entry deve essere una stringa singola
                                if isinstance(entry_val, str):
                                    print(f"  ✓ 'entry' is a string (correct)")
                                else:
                                    print(f"  ✗ 'entry' is NOT a string (PROBLEM!)")
                    
                except json.JSONDecodeError as e:
                    invalid_lines += 1
                    if invalid_lines <= 5:  # Mostra solo i primi 5 errori
                        print(f"[ERRORE] Line {i+1}: {e}")
                        print(f"  Content: {line[:100]}...")
                
                # Limita il controllo per file molto grandi
                if i >= 1000:
                    print(f"\n(Checked first 1000 lines only)")
                    break
        
        print(f"\nSummary:")
        print(f"  Total lines: {total_lines}")
        print(f"  Valid JSON lines: {valid_lines}")
        print(f"  Invalid lines: {invalid_lines}")
        
        if valid_lines > 0:
            print(f"  ✓ File appears valid")
            return True
        else:
            print(f"  ✗ No valid JSON found")
            return False
            
    except Exception as e:
        print(f"[ERRORE] Reading file: {e}")
        return False


def diagnose_ember_dataset(data_dir):
    """Diagnostica completa del dataset EMBER"""
    print("="*80)
    print("EMBER Dataset Diagnostics")
    print("="*80)
    print(f"\nDirectory: {data_dir}\n")
    
    if not os.path.exists(data_dir):
        print(f"[ERRORE] Directory non trovata: {data_dir}")
        return False
    
    # Trova tutti i file JSONL
    jsonl_files = sorted(Path(data_dir).glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"[ERRORE] Nessun file JSONL trovato in {data_dir}")
        return False
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f.name}")
    
    # Controlla ogni file
    all_valid = True
    for jsonl_file in jsonl_files:
        is_valid = check_jsonl_file(jsonl_file)
        all_valid = all_valid and is_valid
    
    print("\n" + "="*80)
    if all_valid:
        print("✓ All files appear valid")
    else:
        print("✗ Some files have issues")
    print("="*80)
    
    return all_valid


def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "dataset/ember2018"
        print(f"No directory specified, using: {data_dir}")
        print(f"Usage: {sys.argv[0]} <path_to_ember_dataset>\n")
    
    diagnose_ember_dataset(data_dir)


if __name__ == "__main__":
    main()