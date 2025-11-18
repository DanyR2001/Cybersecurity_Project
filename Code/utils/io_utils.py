#!/usr/bin/env python3
"""
Modulo per training e validazione dei modelli
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from network.model import EmberMalwareNet
from utils.metrics import MetricsCalculator


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

    # Crea DataLoader
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

    # Inizializza modello
    input_dim = X_train.shape[1]
    model = EmberMalwareNet(input_dim=input_dim, dropout_rate=dropout_rate).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_acc = 0.0
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            status = "[BEST]"
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            status = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} {status}")

    # Valutazione finale
    print("\nValutazione finale su test set...")
    test_metrics, _, _, _ = MetricsCalculator.evaluate_model(model, test_loader, device)
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

    # Copia array per evitare warning NumPy non-writable
    X_test_copy = X_test.copy() if hasattr(X_test, 'copy') else X_test
    y_test_copy = y_test.copy() if hasattr(y_test, 'copy') else y_test
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_copy),
        torch.FloatTensor(y_test_copy)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EmberMalwareNet(input_dim=input_dim, dropout_rate=0.2).to(device)
    model.load_state_dict(checkpoint)

    print(f"Valutazione modello {name}...")
    test_metrics, _, _, _ = MetricsCalculator.evaluate_model(model, test_loader, device)
    MetricsCalculator.print_metrics(test_metrics, f"Test Metrics - {name}")

    return model, test_metrics

def save_results_json(results, save_path="experiment_results.json"):
    """Salva risultati in formato JSON"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in: {save_path}")


def check_models_exist(models={
        'clean': 'model_clean.pth',
        'poisoned': 'model_poisoned.pth',
        'noisy': 'model_noisy.pth'
    }):
    """Verifica esistenza dei modelli"""
    existing = {name: os.path.exists(path) for name, path in models.items()}
    return existing

def load_or_create_poison_indices(config, X_train, y_train):
    """
    Carica poison_indices esistente o crea nuovo dataset avvelenato.
    Risolve il warning dei poison_indices mancanti.
    
    Returns:
        X_train_poisoned, y_train_poisoned, poison_indices, poisoning_info
    """
    if os.path.exists(config.POISON_INDICES_PATH):
        # Carica indici esistenti
        poison_indices = np.load(config.POISON_INDICES_PATH).tolist()
        print(f"\n[+] Poison indices caricati: {len(poison_indices)} campioni")
        
        # Ricrea dataset avvelenato usando gli stessi indici
        X_train_poisoned, y_train_poisoned, _ = poison_dataset(
            X_train, y_train,
            poison_rate=config.POISON_RATE,
            target_label=1,
            flip_to_label=0,
            balanced=config.BALANCED_POISONING
        )
        
        poisoning_info = {
            'total_samples': len(y_train),
            'poisoned_samples': len(poison_indices),
            'poison_rate': config.POISON_RATE,
            'balanced_poisoning': config.BALANCED_POISONING,
            'attack_type': config.ATTACK_TYPE,
            'poison_indices': poison_indices
        }
    else:
        # Crea nuovo dataset avvelenato
        print("\n[*] Creazione nuovo dataset avvelenato...")
        X_train_poisoned, y_train_poisoned, poison_indices = poison_dataset(
            X_train, y_train,
            poison_rate=config.POISON_RATE,
            target_label=1,
            flip_to_label=0,
            balanced=config.BALANCED_POISONING
        )
        
        # Salva gli indici
        np.save(config.POISON_INDICES_PATH, np.array(poison_indices))
        print(f"[+] Poison indices salvati in {config.POISON_INDICES_PATH}")
        
        poisoning_info = {
            'total_samples': len(y_train),
            'poisoned_samples': len(poison_indices),
            'poison_rate': config.POISON_RATE,
            'balanced_poisoning': config.BALANCED_POISONING,
            'attack_type': config.ATTACK_TYPE,
            'poison_indices': poison_indices
        }
    
    return X_train_poisoned, y_train_poisoned, poison_indices, poisoning_info
