#!/usr/bin/env python3
"""
Modulo per caricamento e preprocessing del dataset EMBER
"""

import os
import json
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif
import ember


def select_features(X_train, y_train, X_test=None, corr_threshold=0.95, mi_top_k=500, verbose=True, sample_size=10000):
    """
    Selezione feature combinata:
      1) Rimuove feature con correlazione assoluta > corr_threshold (pearson)
         MANTIENE la feature con varianza maggiore tra le due correlate
      2) (Opzionale) calcola mutual information e mantiene top-k feature

    Args:
        X_train: numpy array or DataFrame (training features)
        y_train: numpy array (labels)
        X_test: numpy array or DataFrame (test features) - opzionale
        corr_threshold: soglia per rimuovere feature altamente correlate (0..1)
        mi_top_k: se non None, mantiene le top-k feature per mutual information
        verbose: stampa info
        sample_size: numero di samples per subset (default 10000) per accelerare

    Returns:
        X_train_sel, X_test_sel, selected_columns (list)
    """
    # Converti in DataFrame se necessario
    if not isinstance(X_train, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train)
    else:
        X_train_df = X_train.copy()

    if X_test is not None and not isinstance(X_test, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test, columns=X_train_df.columns)
    elif X_test is not None:
        X_test_df = X_test.copy()
    else:
        X_test_df = None

    n_features_before = X_train_df.shape[1]
    if verbose:
        print(f"Feature selection: inizio con {n_features_before} feature")

    # Sampling globale: crea un subset per tutta la selezione (per coerenza)
    actual_sample_size = min(sample_size, len(X_train_df))
    if verbose:
        print(f"  Uso subset di {actual_sample_size} samples per accelerare la feature selection")
    X_train_sample_df = X_train_df.sample(n=actual_sample_size, random_state=42)
    y_train_sample = y_train[X_train_sample_df.index]  # Allinea y al sample
    X_train_sample = X_train_sample_df.values  # NumPy per velocità

    # Fix: Rimuovi features con varianza zero dal sample (evita divide by zero in corrcoef)
    variances = np.var(X_train_sample, axis=0)
    non_constant_mask = variances > 0
    constant_count = np.sum(~non_constant_mask)
    if verbose and constant_count > 0:
        print(f"  Rimossi {constant_count} features costanti (var=0) per evitare warnings/errore in correlazione")

    X_train_sample = X_train_sample[:, non_constant_mask]
    sample_columns = X_train_sample_df.columns[non_constant_mask]  # Track columns dopo filtro costanti

    # 1) Correlation-based removal (Pearson) con joblib
    def compute_corr(i, j):
        if i < j:
            corr_val = np.corrcoef(X_train_sample[:, i], X_train_sample[:, j])[0, 1]
            return (i, j, abs(corr_val))
        return None

    n_features = X_train_sample.shape[1]
    pairs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]

    # Parallelizza con joblib
    if verbose:
        print(f"  Calcolo correlazioni tra {len(pairs):,} coppie di features...")
    
    corr_results = Parallel(n_jobs=-1)(
        delayed(compute_corr)(i, j) for i, j in pairs
    )
    corr_results = [r for r in corr_results if r is not None]

    # Costruisci to_drop: MANTIENE la feature con varianza maggiore, DROP l'altra
    to_drop = set()
    kept_due_to_higher_var = set()
    
    for i, j, corr_val in corr_results:
        if corr_val > corr_threshold:
            var_i = np.var(X_train_sample[:, i])
            var_j = np.var(X_train_sample[:, j])
            
            # DROP la feature con varianza MINORE
            if var_i < var_j:
                to_drop.add(sample_columns[i])
                kept_due_to_higher_var.add(sample_columns[j])
            else:
                to_drop.add(sample_columns[j])
                kept_due_to_higher_var.add(sample_columns[i])

    if verbose:
        high_corr_pairs = len([r for r in corr_results if r[2] > corr_threshold])
        print(f"  Trovate {high_corr_pairs} coppie con correlazione > {corr_threshold}")
        print(f"  Rimuovo {len(to_drop)} feature (mantenendo quelle con varianza maggiore)")
        print(f"  Mantengo {len(kept_due_to_higher_var)} feature con alta correlazione (varianza maggiore)")

    # Selected cols: tutte le non-costanti TRANNE quelle in to_drop
    selected_cols = [c for c in sample_columns if c not in to_drop]
    X_train_reduced = X_train_df[selected_cols]

    if verbose:
        print(f"  Dopo rimozione correlazioni: {len(selected_cols)} features")

    # 2) Mutual Information selection (opzionale) con n_jobs
    if mi_top_k is not None:
        if len(selected_cols) > mi_top_k:
            if verbose:
                print(f"  Calcolo Mutual Information e seleziono top-{mi_top_k} feature")
            
            # Usa sample per MI (già allineato)
            X_train_reduced_sample = X_train_sample_df[selected_cols]
            mi = mutual_info_classif(
                X_train_reduced_sample.values, 
                y_train_sample, 
                discrete_features=False, 
                random_state=42, 
                n_jobs=-1
            )
            mi_series = pd.Series(mi, index=selected_cols)
            mi_top = mi_series.sort_values(ascending=False).head(mi_top_k).index.tolist()
            selected_cols = mi_top
            X_train_reduced = X_train_reduced[selected_cols]
            
            if verbose:
                print(f"  Dopo Mutual Information: {len(selected_cols)} features")
        else:
            if verbose:
                print(f"  Skippo MI: ho già solo {len(selected_cols)} features (<= {mi_top_k})")

    # Applica stessa selezione a X_test
    if X_test_df is not None:
        X_test_reduced = X_test_df[selected_cols]
    else:
        X_test_reduced = None

    n_features_after = X_train_reduced.shape[1]
    if verbose:
        print(f"Feature selection: risultato FINALE {n_features_after} feature (da {n_features_before})")
        reduction_pct = (1 - n_features_after/n_features_before) * 100
        print(f"  Riduzione: {reduction_pct:.1f}%")

    return X_train_reduced.values, None if X_test_reduced is None else X_test_reduced.values, list(selected_cols)


def load_and_prepare_data(data_dir, batch_size=256, load_train=True,
                          corr_threshold=0.95, mi_top_k=500, save_selected_features=True,
                          sample_size=10000): 
    """
    Carica e prepara i dati EMBER. Applica feature selection se richiesto.
    
    Args:
        data_dir: directory contenente il dataset EMBER
        batch_size: dimensione batch per DataLoader
        load_train: se True carica anche training set
        corr_threshold: soglia correlazione per feature selection
        mi_top_k: numero di top feature da mantenere (mutual information)
        save_selected_features: salva lista feature selezionate
        sample_size: samples per subset nella feature selection
    
    Returns:
        X_train, y_train, X_test, y_test (X_train e y_train possono essere None se load_train=False)
    """
    print(f"\n=== Caricamento Dataset da {data_dir} ===")
    
    selected_json = "selected_features.json"

    if load_train:
        X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
        X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")

        # Rimuovi campioni unlabeled
        train_mask = y_train != -1
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        test_mask = y_test != -1
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Benign: {int(np.sum(y_train == 0))}")
        print(f"  Malware: {int(np.sum(y_train == 1))}")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  Benign: {int(np.sum(y_test == 0))}")
        print(f"  Malware: {int(np.sum(y_test == 1))}")

        # Check per caricare da JSON se esiste
        if os.path.exists(selected_json) and (corr_threshold is not None or mi_top_k is not None):
            with open(selected_json, "r") as f:
                selected_data = json.load(f)
                selected_cols = selected_data["selected_indices_or_names"]
            print(f"Carico {len(selected_cols)} feature selezionate da {selected_json} (skippo selezione)")
            return X_train[:, selected_cols], y_train, X_test[:, selected_cols], y_test

        # Feature selection se non caricata
        if corr_threshold is not None or mi_top_k is not None:
            X_train_sel, X_test_sel, selected_cols = select_features(
                X_train, y_train, X_test,
                corr_threshold=corr_threshold,
                mi_top_k=mi_top_k,
                verbose=True,
                sample_size=sample_size
            )
            if save_selected_features:
                with open(selected_json, "w") as f:
                    json.dump({
                        "selected_columns_count": len(selected_cols), 
                        "selected_indices_or_names": selected_cols,
                        "config": {
                            "corr_threshold": corr_threshold,
                            "mi_top_k": mi_top_k,
                            "sample_size": sample_size
                        }
                    }, f, indent=2)
                print(f"Selected feature list salvata in {selected_json}")
            return X_train_sel, y_train, X_test_sel, y_test
        else:
            return X_train, y_train, X_test, y_test
    else:
        print("Caricamento solo test set")
        X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")
        test_mask = y_test != -1
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  Benign: {int(np.sum(y_test == 0))}")
        print(f"  Malware: {int(np.sum(y_test == 1))}")
        
        # Applica feature selection se JSON esiste
        if os.path.exists(selected_json):
            with open(selected_json, "r") as f:
                selected_data = json.load(f)
                selected_cols = selected_data["selected_indices_or_names"]
            print(f"Applicando feature selection: {len(selected_cols)} features (da {selected_json})")
            X_test = X_test[:, selected_cols]
        elif corr_threshold is not None or mi_top_k is not None:
            print(f"[WARNING] Feature selection richiesta ma {selected_json} non trovato!")
            print("         Usando tutte le features originali. I modelli potrebbero non caricarsi.")
        
        return None, None, X_test, y_test


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=256):
    """
    Crea DataLoader PyTorch per training e test
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        batch_size: dimensione batch
        
    Returns:
        train_loader, test_loader
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader