#!/usr/bin/env python3
"""
Esperimento completo: Training pulito -> Poisoning -> Rumore Gaussiano
Con calcolo metriche e visualizzazione comparativa
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import ember
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import json
from datetime import datetime

from model import EmberMalwareNet, train_model, validate_model


class MetricsCalculator:
    """Calcola e salva tutte le metriche di valutazione"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calcola tutte le metriche principali"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    @staticmethod
    def evaluate_model(model, data_loader, device):
        """Valuta il modello e restituisce predizioni e metriche"""
        model.eval()
        all_targets = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating", leave=False):
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.sigmoid(output).squeeze()
                preds = (probs > 0.5).float()
                
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = MetricsCalculator.calculate_metrics(
            all_targets, all_preds, all_probs
        )
        
        return metrics, all_targets, all_preds, all_probs


def poison_dataset(X, y, poison_rate=0.1, target_label=0, flip_to_label=1):
    """
    Avvelena il dataset flippando le label.
    
    Args:
        X: Features
        y: Labels
        poison_rate: Percentuale di campioni da avvelenare (0.0-1.0)
        target_label: Label da avvelenare (default: 0 = benign)
        flip_to_label: Label a cui flippare (default: 1 = malware)
    
    Returns:
        X_poisoned, y_poisoned, poison_indices
    """
    print(f"\n=== Poisoning Dataset ===")
    print(f"Poison rate: {poison_rate*100:.1f}%")
    print(f"Target label: {target_label} -> {flip_to_label}")
    
    # Trova gli indici dei campioni target
    target_indices = np.where(y == target_label)[0]
    n_poison = int(len(target_indices) * poison_rate)
    
    # Seleziona casualmente i campioni da avvelenare
    np.random.seed(42)
    poison_indices = np.random.choice(target_indices, size=n_poison, replace=False)
    
    # Crea copie per evitare modifiche inplace
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    
    # Flippa le label
    y_poisoned[poison_indices] = flip_to_label
    
    print(f"Samples poisoned: {n_poison} / {len(target_indices)}")
    print(f"Poisoned indices: {len(poison_indices)}")
    
    return X_poisoned, y_poisoned, poison_indices


def add_gaussian_noise_to_model(model, std=0.01):
    """
    Aggiunge rumore gaussiano ai pesi del modello.
    
    Args:
        model: Modello PyTorch
        std: Deviazione standard del rumore gaussiano
    """
    print(f"\n=== Aggiunta Rumore Gaussiano ai Pesi ===")
    print(f"Standard deviation: {std}")
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:  # Solo sui pesi, non sui bias
                noise = torch.randn_like(param) * std
                param.add_(noise)
                print(f"  Noise added to: {name} (shape: {param.shape})")
    
    print("[OK] Rumore aggiunto a tutti i layer")


def plot_comparison(results, save_path="comparison_plot.png"):
    """
    Crea grafici comparativi per i tre scenari.
    
    Args:
        results: Dict con metriche per clean, poisoned, noisy
        save_path: Path dove salvare il grafico
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    scenarios = ['Clean', 'Poisoned', 'Poisoned + Noise']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparison: Clean vs Poisoned vs Poisoned+Noise', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar chart metriche principali
    ax = axes[0, 0]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    clean_vals = [results['clean']['test'][m] for m in metrics_to_plot]
    poison_vals = [results['poisoned']['test'][m] for m in metrics_to_plot]
    noise_vals = [results['noisy']['test'][m] for m in metrics_to_plot]
    
    ax.bar(x - width, clean_vals, width, label='Clean', color='green', alpha=0.8)
    ax.bar(x, poison_vals, width, label='Poisoned', color='orange', alpha=0.8)
    ax.bar(x + width, noise_vals, width, label='Poisoned+Noise', color='red', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Main Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot], rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 2: Accuracy drop
    ax = axes[0, 1]
    scenarios_short = ['Clean', 'Poisoned', 'P+Noise']
    accuracies = [
        results['clean']['test']['accuracy'],
        results['poisoned']['test']['accuracy'],
        results['noisy']['test']['accuracy']
    ]
    colors = ['green', 'orange', 'red']
    bars = ax.bar(scenarios_short, accuracies, color=colors, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Degradation')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: F1-Score comparison
    ax = axes[0, 2]
    f1_scores = [
        results['clean']['test']['f1_score'],
        results['poisoned']['test']['f1_score'],
        results['noisy']['test']['f1_score']
    ]
    bars = ax.bar(scenarios_short, f1_scores, color=colors, alpha=0.8)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Comparison')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Confusion Matrix - Clean
    ax = axes[1, 0]
    cm_clean = np.array([
        [results['clean']['test']['true_negative'], results['clean']['test']['false_positive']],
        [results['clean']['test']['false_negative'], results['clean']['test']['true_positive']]
    ])
    im = ax.imshow(cm_clean, cmap='Greens', alpha=0.8)
    ax.set_title('Confusion Matrix - Clean')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malware'])
    ax.set_yticklabels(['Benign', 'Malware'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm_clean[i, j], ha="center", va="center", color="black", fontsize=12)
    
    # Plot 5: Confusion Matrix - Poisoned
    ax = axes[1, 1]
    cm_poison = np.array([
        [results['poisoned']['test']['true_negative'], results['poisoned']['test']['false_positive']],
        [results['poisoned']['test']['false_negative'], results['poisoned']['test']['true_positive']]
    ])
    im = ax.imshow(cm_poison, cmap='Oranges', alpha=0.8)
    ax.set_title('Confusion Matrix - Poisoned')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malware'])
    ax.set_yticklabels(['Benign', 'Malware'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm_poison[i, j], ha="center", va="center", color="black", fontsize=12)
    
    # Plot 6: Confusion Matrix - Noisy
    ax = axes[1, 2]
    cm_noise = np.array([
        [results['noisy']['test']['true_negative'], results['noisy']['test']['false_positive']],
        [results['noisy']['test']['false_negative'], results['noisy']['test']['true_positive']]
    ])
    im = ax.imshow(cm_noise, cmap='Reds', alpha=0.8)
    ax.set_title('Confusion Matrix - Poisoned+Noise')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malware'])
    ax.set_yticklabels(['Benign', 'Malware'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm_noise[i, j], ha="center", va="center", color="black", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico salvato in: {save_path}")
    plt.close()


def save_results_json(results, save_path="results.json"):
    """Salva i risultati in formato JSON"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Risultati salvati in: {save_path}")


def load_and_prepare_data(data_dir, batch_size=256):
    """Carica e prepara i dati EMBER"""
    print(f"\n=== Caricamento Dataset da {data_dir} ===")
    
    # Carica dataset
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
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, device, 
                       epochs=30, batch_size=256, lr=0.001, name="model"):
    """Training completo e valutazione"""
    
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    # Prepara DataLoader
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
    
    # Crea modello
    input_dim = X_train.shape[1]
    model = EmberMalwareNet(input_dim=input_dim, dropout_rate=0.2).to(device)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\nTraining per {epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            status = "[BEST]"
        else:
            status = ""
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} {status}")
    
    # Valutazione finale
    print(f"\nValutazione finale su test set...")
    test_metrics, _, _, _ = MetricsCalculator.evaluate_model(model, test_loader, device)
    
    print(f"\n=== Metriche Test - {name} ===")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    return model, test_metrics


def main():
    # Configurazione
    DATA_DIR = "dataset/ember_dataset_2018_2"
    EPOCHS = 30
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    POISON_RATE = 0.1  # 10% di poisoning
    NOISE_STD = 0.01   # Deviazione standard rumore gaussiano
    
    # Parsing argomenti
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    
    if not os.path.exists(DATA_DIR):
        print(f"ERRORE: Directory non trovata: {DATA_DIR}")
        sys.exit(1)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n[DEVICE] Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[DEVICE] Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"\n[DEVICE] Using CPU")
    
    # Carica dati
    X_train, y_train, X_test, y_test = load_and_prepare_data(DATA_DIR, BATCH_SIZE)
    
    # Dizionario per salvare tutti i risultati
    results = {
        'experiment_date': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'poison_rate': POISON_RATE,
            'noise_std': NOISE_STD,
        }
    }
    
    # ========================================
    # ESPERIMENTO 1: DATASET PULITO
    # ========================================
    print("\n" + "="*80)
    print("ESPERIMENTO 1: TRAINING SU DATASET PULITO")
    print("="*80)
    
    model_clean, metrics_clean = train_and_evaluate(
        X_train, y_train, X_test, y_test, device,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
        name="Clean Dataset"
    )
    
    results['clean'] = {'test': metrics_clean}
    torch.save(model_clean.state_dict(), "model_clean.pth")
    
    # ========================================
    # ESPERIMENTO 2: DATASET AVVELENATO
    # ========================================
    print("\n" + "="*80)
    print("ESPERIMENTO 2: TRAINING SU DATASET AVVELENATO")
    print("="*80)
    
    X_train_poisoned, y_train_poisoned, poison_indices = poison_dataset(
        X_train, y_train, 
        poison_rate=POISON_RATE,
        target_label=0,  # Avvelena campioni benign
        flip_to_label=1  # Li fa sembrare malware
    )
    
    model_poisoned, metrics_poisoned = train_and_evaluate(
        X_train_poisoned, y_train_poisoned, X_test, y_test, device,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
        name="Poisoned Dataset"
    )
    
    results['poisoned'] = {'test': metrics_poisoned}
    torch.save(model_poisoned.state_dict(), "model_poisoned.pth")
    
    # ========================================
    # ESPERIMENTO 3: MODELLO AVVELENATO + RUMORE
    # ========================================
    print("\n" + "="*80)
    print("ESPERIMENTO 3: MODELLO AVVELENATO + RUMORE GAUSSIANO")
    print("="*80)
    
    # Carica il modello avvelenato (facciamo una copia)
    model_noisy = EmberMalwareNet(input_dim=X_train.shape[1], dropout_rate=0.2).to(device)
    model_noisy.load_state_dict(torch.load("model_poisoned.pth"))
    
    # Aggiungi rumore gaussiano
    add_gaussian_noise_to_model(model_noisy, std=NOISE_STD)
    
    # Valuta il modello con rumore
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nValutazione modello con rumore...")
    metrics_noisy, _, _, _ = MetricsCalculator.evaluate_model(model_noisy, test_loader, device)
    
    print(f"\n=== Metriche Test - Poisoned + Noise ===")
    for key, value in metrics_noisy.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    results['noisy'] = {'test': metrics_noisy}
    torch.save(model_noisy.state_dict(), "model_noisy.pth")
    
    # ========================================
    # VISUALIZZAZIONE E SALVATAGGIO
    # ========================================
    print("\n" + "="*80)
    print("GENERAZIONE REPORT E GRAFICI")
    print("="*80)
    
    # Salva risultati JSON
    save_results_json(results, "experiment_results.json")
    
    # Crea grafici comparativi
    plot_comparison(results, "comparison_plot.png")
    
    # Stampa summary finale
    print("\n" + "="*80)
    print("SUMMARY FINALE")
    print("="*80)
    print("\nAccuracy Comparison:")
    print(f"  Clean Model:          {results['clean']['test']['accuracy']:.4f}")
    print(f"  Poisoned Model:       {results['poisoned']['test']['accuracy']:.4f}")
    print(f"  Poisoned + Noise:     {results['noisy']['test']['accuracy']:.4f}")
    
    print("\nF1-Score Comparison:")
    print(f"  Clean Model:          {results['clean']['test']['f1_score']:.4f}")
    print(f"  Poisoned Model:       {results['poisoned']['test']['f1_score']:.4f}")
    print(f"  Poisoned + Noise:     {results['noisy']['test']['f1_score']:.4f}")
    
    acc_drop_poison = results['clean']['test']['accuracy'] - results['poisoned']['test']['accuracy']
    acc_drop_noise = results['poisoned']['test']['accuracy'] - results['noisy']['test']['accuracy']
    
    print("\nAccuracy Drop:")
    print(f"  Clean -> Poisoned:    {acc_drop_poison:.4f} ({acc_drop_poison*100:.2f}%)")
    print(f"  Poisoned -> Noise:    {acc_drop_noise:.4f} ({acc_drop_noise*100:.2f}%)")
    
    print("\nFile salvati:")
    print("  - model_clean.pth")
    print("  - model_poisoned.pth")
    print("  - model_noisy.pth")
    print("  - experiment_results.json")
    print("  - comparison_plot.png")
    
    print("\n" + "="*80)
    print("ESPERIMENTO COMPLETATO!")
    print("="*80)


if __name__ == "__main__":
    main()