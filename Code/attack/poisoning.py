#!/usr/bin/env python3
"""
Modulo per attacchi di poisoning e perturbazione del modello
"""

import numpy as np
import torch


def poison_dataset(X, y, poison_rate=0.1, target_label=1, flip_to_label=0):
    """
    Attacco: label flipping su alcuni malware (malware -> benign)
    
    Args:
        X: feature array
        y: label array
        poison_rate: percentuale di campioni da avvelenare
        target_label: etichetta da targetizzare (default: 1 = malware)
        flip_to_label: nuova etichetta (default: 0 = benign)
    
    Returns:
        X_poisoned (deep copy), y_poisoned, poison_indices
    """
    print("\n=== Poisoning Dataset ===")
    print("ATTACCO: Malware -> Benign (label flipping)")
    print(f"Poison rate: {poison_rate*100:.1f}%")

    target_indices = np.where(y == target_label)[0]
    n_poison = int(len(target_indices) * poison_rate)

    np.random.seed(42)
    poison_indices = np.random.choice(target_indices, size=n_poison, replace=False)

    X_poisoned = X.copy()
    y_poisoned = y.copy()

    y_poisoned[poison_indices] = flip_to_label

    print("Poisoning statistics:")
    print(f"  Malware totali: {len(target_indices)}")
    print(f"  Malware avvelenati: {n_poison}")
    print(f"  Distribuzione finale: Benign={int(np.sum(y_poisoned == 0))}, Malware={int(np.sum(y_poisoned == 1))}")

    return X_poisoned, y_poisoned, poison_indices.tolist()


def add_gaussian_noise_to_model(model, std=0.01, target_layers=None):
    """
    Aggiunge rumore gaussiano ai parametri del modello.
    
    Args:
        model: modello PyTorch
        std: deviazione standard del rumore gaussiano
        target_layers: lista di nomi layer da perturbare (default: tutti i pesi)
    
    Returns:
        noise_stats: dizionario con statistiche del rumore (JSON serializable)
    """
    print("\n=== Aggiunta Rumore Gaussiano ===")
    print(f"Standard deviation: {std}")

    total_params = 0
    noisy_params = 0
    noise_stats = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            total_params += 1
            add_noise = False

            if target_layers is not None:
                add_noise = any(layer_name in name for layer_name in target_layers)
            else:
                add_noise = 'weight' in name  # applica ai pesi

            if add_noise and param.requires_grad:
                original_mean = float(param.mean().cpu().item())
                original_std = float(param.std().cpu().item())

                noise = torch.randn_like(param) * float(std)
                param.add_(noise)
                noisy_params += 1

                new_mean = float(param.mean().cpu().item())
                new_std = float(param.std().cpu().item())

                noise_stats[name] = {
                    'shape': list(param.shape),
                    'noise_std': float(std),
                    'mean_shift': float(new_mean - original_mean),
                    'std_change': float(new_std - original_std),
                    'original_mean': original_mean,
                    'original_std': original_std,
                    'new_mean': new_mean,
                    'new_std': new_std
                }

                print(f"  Modified {name:30s} | shape: {str(param.shape):15s} | Δμ: {new_mean - original_mean:+.6f} | Δσ: {new_std - original_std:+.6f}")

    print(f"Param tensors visitati: {total_params}, param tensors alterati: {noisy_params}")
    return noise_stats


def backdoor_attack(X, y, trigger_pattern, target_label=0, backdoor_rate=0.05):
    """
    Attacco backdoor: inserisce un pattern trigger in alcuni campioni
    e cambia la loro etichetta
    
    Args:
        X: feature array
        y: label array
        trigger_pattern: pattern da inserire (array o valore scalare)
        target_label: etichetta target per backdoor
        backdoor_rate: percentuale di campioni da backdoorare
    
    Returns:
        X_backdoor, y_backdoor, backdoor_indices
    """
    print("\n=== Backdoor Attack ===")
    print(f"Backdoor rate: {backdoor_rate*100:.1f}%")
    
    n_backdoor = int(len(X) * backdoor_rate)
    np.random.seed(42)
    backdoor_indices = np.random.choice(len(X), size=n_backdoor, replace=False)
    
    X_backdoor = X.copy()
    y_backdoor = y.copy()
    
    # Inserisci trigger pattern (esempio: setta alcune feature a valori fissi)
    if np.isscalar(trigger_pattern):
        X_backdoor[backdoor_indices, :10] = trigger_pattern  # trigger nelle prime 10 feature
    else:
        X_backdoor[backdoor_indices, :len(trigger_pattern)] = trigger_pattern
    
    y_backdoor[backdoor_indices] = target_label
    
    print(f"  Campioni backdoorati: {n_backdoor}")
    print(f"  Target label: {target_label}")
    
    return X_backdoor, y_backdoor, backdoor_indices.tolist()