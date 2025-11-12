#!/usr/bin/env python3
"""
Esperimento completo: Training pulito -> Poisoning -> Rumore Gaussiano
Con calcolo metriche e visualizzazione comparativa

Struttura modulare:
- preprocessing/: caricamento e preparazione dati
- attack/: attacchi di poisoning e perturbazione
- network/: definizione modello e training
- utils/: metriche, visualizzazione, I/O
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch

# Import moduli personalizzati
from preprocessing.data_loader import load_and_prepare_data
from attack.poisoning import poison_dataset, add_gaussian_noise_to_model
from network.trainer import train_and_evaluate, load_and_evaluate
from utils.metrics import MetricsCalculator
from utils.visualization import plot_comparison
from utils.io_utils import save_results_json, check_models_exist


class ExperimentConfig:
    """Configurazione centralizzata dell'esperimento"""
    
    def __init__(self):
        # Percorsi
        self.DATA_DIR = "dataset/ember_dataset_2018_2"
        
        # Parametri training
        self.EPOCHS = 10
        self.BATCH_SIZE = 256
        self.LEARNING_RATE = 0.001
        self.DROPOUT_RATE = 0.5
        self.WEIGHT_DECAY = 1e-5
        
        # Parametri attacco
        self.POISON_RATE = 0.2  # 20% di poisoning
        self.NOISE_STD = 0.0001
        self.ATTACK_TYPE = 'Label Flipping (Malware->Benign)'
        
        # Feature selection
        self.CORR_THRESHOLD = 0.98   # rimuove feature con correlazione > 0.98
        self.MI_TOP_K = 40         # se settato (es. 500) mantiene top-k MI
        
        # Percorsi modelli
        self.MODEL_PATHS = {
            'clean': 'model_clean.pth',
            'poisoned': 'model_poisoned.pth',
            'noisy': 'model_noisy.pth'
        }
    
    def to_dict(self):
        """Converte configurazione in dizionario per serializzazione"""
        return {
            'data_dir': self.DATA_DIR,
            'epochs': self.EPOCHS,
            'batch_size': self.BATCH_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'dropout_rate': self.DROPOUT_RATE,
            'weight_decay': self.WEIGHT_DECAY,
            'poison_rate': self.POISON_RATE,
            'noise_std': self.NOISE_STD,
            'attack_type': self.ATTACK_TYPE,
            'corr_threshold': self.CORR_THRESHOLD,
            'mi_top_k': self.MI_TOP_K
        }
    
    def update_from_args(self, args):
        """Aggiorna configurazione da argomenti command line"""
        if len(args) > 1:
            self.DATA_DIR = args[1]


def setup_device():
    """Configura e restituisce il device PyTorch appropriato"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("DEVICE: Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("DEVICE: Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("DEVICE: Using CPU")
    return device


def experiment_clean_model(config, X_train, y_train, X_test, y_test, device, force_retrain=False):
    """
    Esperimento 1: Training su dataset pulito
    
    Returns:
        model, metrics
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO 1: DATASET PULITO")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['clean']
    
    if os.path.exists(model_path) and not force_retrain:
        model, metrics = load_and_evaluate(
            model_path, X_test, y_test, device, 
            config.BATCH_SIZE, "Clean Dataset"
        )
    else:
        print("Training modello clean da zero...")
        model, metrics, _ = train_and_evaluate(
            X_train, y_train, X_test, y_test, device,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE,
            dropout_rate=config.DROPOUT_RATE,
            weight_decay=config.WEIGHT_DECAY,
            name="Clean Dataset",
            save_path=model_path
        )
        print(f"Modello salvato: {model_path}")
    
    return model, metrics


def experiment_poisoned_model(config, X_train, y_train, X_test, y_test, device, force_retrain=False):
    """
    Esperimento 2: Training su dataset avvelenato
    
    Returns:
        model, metrics, poisoning_info
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO 2: DATASET AVVELENATO")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['poisoned']
    
    if os.path.exists(model_path) and not force_retrain:
        model, metrics = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Poisoned Dataset"
        )
        poisoning_info = None  # Non abbiamo info se carichiamo modello esistente
    else:
        print("Creazione dataset avvelenato e training...")
        X_train_poisoned, y_train_poisoned, poison_indices = poison_dataset(
            X_train, y_train,
            poison_rate=config.POISON_RATE,
            target_label=1,
            flip_to_label=0
        )
        
        poisoning_info = {
            'total_malware': int(np.sum(y_train == 1)),
            'poisoned_samples': len(poison_indices),
            'poison_rate': config.POISON_RATE,
            'attack_type': config.ATTACK_TYPE,
            'poison_indices_sample': poison_indices[:100]
        }
        
        model, metrics, _ = train_and_evaluate(
            X_train_poisoned, y_train_poisoned, X_test, y_test, device,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE,
            dropout_rate=config.DROPOUT_RATE,
            weight_decay=config.WEIGHT_DECAY,
            name="Poisoned Dataset",
            save_path=model_path
        )
        print(f"Modello salvato: {model_path}")
    
    return model, metrics, poisoning_info


def experiment_noisy_model(config, X_test, y_test, device, force_retrain=False):
    """
    Esperimento 3: Modello avvelenato + rumore gaussiano
    
    Returns:
        model, metrics, noise_stats
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO 3: MODELLO AVVELENATO + RUMORE")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['noisy']
    poisoned_path = config.MODEL_PATHS['poisoned']
    
    if not os.path.exists(poisoned_path):
        raise FileNotFoundError(
            f"Modello poisoned non trovato: {poisoned_path}. "
            "Eseguire prima esperimento 2."
        )
    
    if os.path.exists(model_path) and not force_retrain:
        model, metrics = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Poisoned+Noise"
        )
        noise_stats = None
    else:
        from network.model import EmberMalwareNet
        from torch.utils.data import DataLoader, TensorDataset
        
        # Carica modello poisoned
        model = EmberMalwareNet(input_dim=X_test.shape[1], dropout_rate=0.2).to(device)
        model.load_state_dict(torch.load(poisoned_path, map_location=device))
        
        # Aggiungi rumore
        noise_stats = add_gaussian_noise_to_model(model, std=config.NOISE_STD)
        
        # Valuta
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        metrics, _, _, _ = MetricsCalculator.evaluate_model(model, test_loader, device)
        MetricsCalculator.print_metrics(metrics, "Test Metrics - Poisoned+Noise")
        
        # Salva
        torch.save(model.state_dict(), model_path)
        print(f"Modello salvato: {model_path}")
    
    return model, metrics, noise_stats


def print_final_summary(results):
    """Stampa sommario finale degli esperimenti"""
    print("\n" + "=" * 80)
    print("SUMMARY FINALE")
    print("=" * 80)
    
    print(f"\nAccuracy Comparison:")
    print(f"  Clean:            {results['clean']['test']['accuracy']:.4f}")
    print(f"  Poisoned:         {results['poisoned']['test']['accuracy']:.4f}")
    print(f"  Poisoned+Noise:   {results['noisy']['test']['accuracy']:.4f}")
    
    print(f"\nF1-Score Comparison:")
    print(f"  Clean:            {results['clean']['test']['f1_score']:.4f}")
    print(f"  Poisoned:         {results['poisoned']['test']['f1_score']:.4f}")
    print(f"  Poisoned+Noise:   {results['noisy']['test']['f1_score']:.4f}")
    
    # Degradazione
    clean_acc = results['clean']['test']['accuracy']
    poison_acc = results['poisoned']['test']['accuracy']
    noisy_acc = results['noisy']['test']['accuracy']
    
    print(f"\nDegradazione Accuracy:")
    print(f"  Poisoning:        {(clean_acc - poison_acc)*100:+.2f}%")
    print(f"  Poisoning+Noise:  {(clean_acc - noisy_acc)*100:+.2f}%")
    
    if 'noise_statistics' in results:
        ns = results['noise_statistics']
        mean_shifts = [s['mean_shift'] for s in ns.values()]
        std_changes = [s['std_change'] for s in ns.values()]
        print(f"\nNoise Statistics:")
        print(f"  Layers altered:      {len(ns)}")
        print(f"  Mean shift (avg):    {np.mean(np.abs(mean_shifts)):.6f}")
        print(f"  Std change (avg):    {np.mean(np.abs(std_changes)):.6f}")
    
    if 'poisoning_info' in results:
        pinfo = results['poisoning_info']
        print(f"\nPoisoning Info:")
        print(f"  Total malware:       {pinfo['total_malware']}")
        print(f"  Poisoned samples:    {pinfo['poisoned_samples']}")
        print(f"  Poison rate:         {pinfo['poison_rate']*100:.1f}%")


def main():
    """Funzione principale che orchestra l'esperimento completo"""
    
    # Configurazione
    config = ExperimentConfig()
    config.update_from_args(sys.argv)
    
    # Verifica directory dataset
    if not os.path.exists(config.DATA_DIR):
        print(f"ERRORE: Directory non trovata: {config.DATA_DIR}")
        sys.exit(1)
    
    # Setup device
    device = setup_device()
    
    # Verifica modelli esistenti
    existing_models = check_models_exist(config.MODEL_PATHS)
    print("\nControllo modelli esistenti:", existing_models)
    need_training = not (existing_models['clean'] and existing_models['poisoned'])
    

    # Carica dati
    if need_training:
        X_train, y_train, X_test, y_test = load_and_prepare_data(
            config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
            load_train=True,
            corr_threshold=config.CORR_THRESHOLD,
            mi_top_k=config.MI_TOP_K,
            save_selected_features=True
        )
    else:
        print("\nCaricamento dati per valutazione (applicando feature selection salvata)...")
        _, _, X_test, y_test = load_and_prepare_data(
            config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
            load_train=False,
            corr_threshold=config.CORR_THRESHOLD,  # Mantieni parametri per consistenza
            mi_top_k=config.MI_TOP_K
        )
        X_train, y_train = None, None
    
    # === MATRICE DI CORRELAZIONE ===
    if X_train is not None:
        from utils.visualization import plot_correlation_matrix
        print("\nGenerazione matrice di correlazione delle feature...")
        plot_correlation_matrix(
            X_train,
            save_path="correlation_matrix.png",
            method='pearson',
            threshold=0.8,
            figsize=(14, 12)
        )
    else:
        print("\nSalto matrice di correlazione: X_train non disponibile (modelli già esistenti).")

    # Inizializza struttura risultati
    results = {
        'experiment_date': datetime.now().isoformat(),
        'config': config.to_dict()
    }
    
    # Esperimento 1: Clean
    _, metrics_clean = experiment_clean_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['clean'] = {'test': metrics_clean}
    
    # Esperimento 2: Poisoned
    _, metrics_poisoned, poisoning_info = experiment_poisoned_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['poisoned'] = {'test': metrics_poisoned}
    if poisoning_info:
        results['poisoning_info'] = poisoning_info
    
    # Esperimento 3: Noisy
    _, metrics_noisy, noise_stats = experiment_noisy_model(
        config, X_test, y_test, device
    )
    results['noisy'] = {'test': metrics_noisy}
    if noise_stats:
        results['noise_statistics'] = noise_stats
    
    # Salva risultati e visualizzazioni
    save_results_json(results, "experiment_results.json")
    plot_comparison(results, "comparison_plot.png")
    
    # Summary finale
    print_final_summary(results)
    
    print("\n" + "=" * 80)
    print("ESPERIMENTO COMPLETATO")
    print("=" * 80)


if __name__ == "__main__":
    main()