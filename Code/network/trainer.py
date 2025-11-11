#!/usr/bin/env python3
"""
Modulo per training e validazione dei modelli
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from network.model import EmberMalwareNet
from utils.metrics import MetricsCalculator
import numpy as np

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Esegue un'epoca di training
    
    Args:
        model: modello PyTorch
        train_loader: DataLoader con dati di training
        criterion: loss function
        optimizer: ottimizzatore
        device: device PyTorch
        
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data = data.to(device).float()
        target = target.to(device).float()

        optimizer.zero_grad()
        logits = model(data)            # logits shape [batch]
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == target.long()).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def validate_epoch(model, val_loader, criterion, device):
    """
    Esegue validazione su un epoch
    
    Args:
        model: modello PyTorch
        val_loader: DataLoader con dati di validazione
        criterion: loss function
        device: device PyTorch
        
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device).float()
            target = target.to(device).float()

            logits = model(data)
            loss = criterion(logits, target)
            total_loss += loss.item() * data.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == target.long()).sum().item()
            total += data.size(0)

    return total_loss / total, correct / total


def train_and_evaluate(X_train, y_train, X_test, y_test, device,
                       epochs=30, batch_size=256, lr=0.001, 
                       dropout_rate=0.5, weight_decay=1e-5,
                       name="model", save_path=None):
    """
    Training completo con validazione e salvataggio
    
    Args:
        X_train, y_train: dati di training
        X_test, y_test: dati di test
        device: device PyTorch
        epochs: numero di epoche
        batch_size: dimensione batch
        lr: learning rate
        dropout_rate: tasso di dropout
        weight_decay: L2 regularization
        name: nome del modello (per logging)
        save_path: percorso dove salvare il modello (opzionale)
        
    Returns:
        model, test_metrics, history
    """
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    # Crea copie writable degli array
    X_train_copy = np.array(X_train, copy=True)
    y_train_copy = np.array(y_train, copy=True)
    X_test_copy = np.array(X_test, copy=True)
    y_test_copy = np.array(y_test, copy=True)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_copy),
        torch.FloatTensor(y_train_copy)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_copy),
        torch.FloatTensor(y_test_copy)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inizializza modello
    input_dim = X_train.shape[1]
    model = EmberMalwareNet(input_dim=input_dim, dropout_rate=dropout_rate).to(device)

    # Calcola class weights per bilanciare il training
    n_benign = np.sum(y_train == 0)
    n_malware = np.sum(y_train == 1)
    pos_weight = torch.tensor([n_benign / n_malware], dtype=torch.float32, device=device)
    
    print(f"\nClass distribution:")
    print(f"  Benign: {n_benign:,} ({n_benign/len(y_train)*100:.1f}%)")
    print(f"  Malware: {n_malware:,} ({n_malware/len(y_train)*100:.1f}%)")
    print(f"  Pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_f1 = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Calcola F1 per early stopping migliore
        temp_metrics, _, _, _ = MetricsCalculator.evaluate_model(model, test_loader, device)
        current_f1 = temp_metrics['f1_score']
        
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            status = f"[BEST F1={current_f1:.4f}]"
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            status = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} {status}")

    # Valutazione finale con threshold optimization
    print("\nValutazione finale su test set...")
    test_metrics, targets, preds, probs = MetricsCalculator.evaluate_model(
        model, test_loader, device
    )
    
    # Trova threshold ottimale per massimizzare F1
    optimal_threshold, optimal_f1 = MetricsCalculator.find_optimal_threshold(
        targets, probs
    )
    
    if optimal_threshold != 0.5:
        print(f"\n  THRESHOLD OPTIMIZATION:")
        print(f"  Default threshold (0.5): F1 = {test_metrics['f1_score']:.4f}")
        print(f"  Optimal threshold ({optimal_threshold:.3f}): F1 = {optimal_f1:.4f}")
        print(f"  Improvement: +{(optimal_f1 - test_metrics['f1_score'])*100:.2f}%")
        
        # Ricalcola metriche con threshold ottimale
        preds_optimal = (probs > optimal_threshold).astype(int)
        test_metrics = MetricsCalculator.calculate_metrics(targets, preds_optimal, probs)
        test_metrics['optimal_threshold'] = float(optimal_threshold)
    
    MetricsCalculator.print_metrics(test_metrics, f"Test Metrics - {name}")

    return model, test_metrics, history


def load_and_evaluate(model_path, X_test, y_test, device, batch_size=256, name="model"):
    """
    Carica un modello salvato e lo valuta
    
    Args:
        model_path: percorso del modello salvato
        X_test, y_test: dati di test
        device: device PyTorch
        batch_size: dimensione batch
        name: nome del modello (per logging)
        
    Returns:
        model, test_metrics
    """
    print(f"\nCaricamento modello esistente: {model_path}")

    # Carica il checkpoint per dedurre input_dim
    checkpoint = torch.load(model_path, map_location=device)
    
    # Deduce input_dim dalla shape di fc1.weight [out_features, in_features]
    if 'fc1.weight' in checkpoint:
        input_dim = checkpoint['fc1.weight'].shape[1]
        print(f"  Detected input_dim: {input_dim} (from saved model)")
    else:
        raise KeyError("Cannot find 'fc1.weight' in checkpoint to deduce input_dim")
    
    # Verifica compatibilità con X_test
    if X_test.shape[1] != input_dim:
        raise ValueError(
            f"X_test ha {X_test.shape[1]} features, ma il modello si aspetta {input_dim} features. "
            "Assicurati di applicare la stessa feature selection usata durante il training."
        )

    # Crea copie writable
    X_test_copy = np.array(X_test, copy=True)
    y_test_copy = np.array(y_test, copy=True)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_copy),
        torch.FloatTensor(y_test_copy)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EmberMalwareNet(input_dim=input_dim, dropout_rate=0.2).to(device)
    model.load_state_dict(checkpoint)

    print(f"Valutazione modello {name}...")
    test_metrics, targets, preds, probs = MetricsCalculator.evaluate_model(
        model, test_loader, device
    )
    
    # Trova threshold ottimale
    optimal_threshold, optimal_f1 = MetricsCalculator.find_optimal_threshold(
        targets, probs
    )
    
    if optimal_threshold != 0.5:
        print(f"\n  THRESHOLD OPTIMIZATION:")
        print(f"  Default threshold (0.5): F1 = {test_metrics['f1_score']:.4f}")
        print(f"  Optimal threshold ({optimal_threshold:.3f}): F1 = {optimal_f1:.4f}")
        
        # Ricalcola con threshold ottimale
        preds_optimal = (probs > optimal_threshold).astype(int)
        test_metrics = MetricsCalculator.calculate_metrics(targets, preds_optimal, probs)
        test_metrics['optimal_threshold'] = float(optimal_threshold)
    
    MetricsCalculator.print_metrics(test_metrics, f"Test Metrics - {name}")

    return model, test_metrics