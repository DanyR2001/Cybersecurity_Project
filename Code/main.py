#!/usr/bin/env python3
"""
Esperimento completo: Backdoor Attack -> Detection -> Defense
Confronto con risultati del paper Severi et al.
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import copy
import json

# Import moduli personalizzati
from preprocessing.data_loader import load_and_prepare_data
from utils.metrics import MetricsCalculator
from utils.visualization import plot_comparison, plot_comparison_enhanced
from utils.metrics import print_final_summary, print_defense_comparison
from utils.io_utils import save_results_json, check_models_exist
from experiment.experiment import (
    experiment_clean_model,
    experiment_backdoor_attack,
    experiment_poison_detection,
    experiment_isolation_forest_defense,
    experiment_pruning_defense,
    experiment_noisy_defense_tuned
)


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
        
        # Parametri attacco BACKDOOR (come nel paper)
        self.POISON_RATE = 0.01  # 1% come nel paper
        self.TRIGGER_SIZE = 128    # 16 - 32 64 - 128 features come nel paper per EmberNN
        self.ATTACK_TYPE = 'Clean-Label Backdoor (SHAP-guided)'
        
        # Parametri DETECTION (Weight Pruning)
        self.DETECTION_PRUNING_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Parametri DEFENSE (Pruning ottimale)
        self.DEFENSE_PRUNING_RATE = None
        
        # Parametri DEFENSE (Noise)
        self.NOISE_STD = 0.03
        
        # Feature selection
        self.CORR_THRESHOLD = 0.98
        self.MI_TOP_K = None  # PIÙ FEATURES per avere spazio per SHAP
        
        # Percorsi modelli
        self.MODEL_PATHS = {
            'clean': 'model_clean.pth',
            'backdoored': 'model_backdoored.pth',
            'pruned': 'model_pruned.pth',
            'noisy': 'model_noisy.pth',
            'isolation_forest': 'model_isolation_forest_defended.pth'
        }
        
        # Percorso poison indices
        self.POISON_INDICES_PATH = 'poison_indices_backdoor.npy'
    
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
            'trigger_size': self.TRIGGER_SIZE,
            'attack_type': self.ATTACK_TYPE,
            'detection_pruning_rates': self.DETECTION_PRUNING_RATES,
            'defense_pruning_rate': self.DEFENSE_PRUNING_RATE,
            'noise_std': self.NOISE_STD,
            'corr_threshold': self.CORR_THRESHOLD,
            'mi_top_k': self.MI_TOP_K
        }
    
    def update_from_args(self, args):
        """Aggiorna configurazione da argomenti command line"""
        if len(args) > 1:
            self.DATA_DIR = args[1]


def setup_device():
    """Configura e restituisce il device PyTorch appropriato"""
    low_memory_mode = False
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  DEVICE: Using Apple Silicon GPU (MPS)")
        low_memory_mode = True  # Attiva automaticamente su Mac
        print("  [LOW MEMORY MODE: Enabled for Apple Silicon]")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("  DEVICE: Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("  DEVICE: Using CPU")
        low_memory_mode = True  # Anche CPU beneficia di low memory
        print("  [LOW MEMORY MODE: Enabled for CPU]")
    
    return device, low_memory_mode


def main():
    """Funzione principale"""
    
    print("\n" + "="*80)
    print(" BACKDOOR POISONING ATTACK & DEFENSE EVALUATION")
    print("   Based on: Severi et al. (USENIX Security 2021)")
    print("="*80)
    
    # Configurazione
    config = ExperimentConfig()
    config.update_from_args(sys.argv)
    
    # Verifica directory
    if not os.path.exists(config.DATA_DIR):
        print(f" ERRORE: Directory non trovata: {config.DATA_DIR}")
        sys.exit(1)
    
    # Setup device
    device, low_memory_mode = setup_device()    
    
    # Carica dati
    print("\n Loading EMBER dataset...")
    X_train, y_train, X_test, y_test = load_and_prepare_data(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        load_train=True,
        corr_threshold=config.CORR_THRESHOLD,
        mi_top_k=config.MI_TOP_K,
        save_selected_features=True
    )
    
    # Struttura risultati
    results = {
        'experiment_date': datetime.now().isoformat(),
        'config': config.to_dict()
    }
    
    # ========================================================================
    # ESPERIMENTO 1: Clean Model (Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print(" PHASE 1: Training Clean Baseline Model")
    print("="*80)
    
    model_clean, metrics_clean = experiment_clean_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['clean'] = {'test': metrics_clean}
    
    # ========================================================================
    # ESPERIMENTO 2: Backdoor Attack (NUOVO!)
    # ========================================================================
    print("\n" + "="*80)
    print(" PHASE 2: Backdoor Attack (SHAP-guided)")
    print("="*80)
    
    model_backdoor, metrics_backdoor, attack_metrics, backdoor, poisoning_info = experiment_backdoor_attack(
        config, model_clean, X_train, y_train, X_test, y_test, device
    )
    
    results['backdoored'] = {
        'test': metrics_backdoor,
        'attack_metrics': attack_metrics
    }
    results['poisoning_info'] = poisoning_info
    
    # ========================================================================
    # ESPERIMENTO 3: Detection (Weight Pruning)
    # ========================================================================
    print("\n" + "="*80)
    print(" PHASE 3: Poisoning Detection (Weight Pruning)")
    print("="*80)
    
    detection_results_to_save, detection_results_full = experiment_poison_detection(
        config, model_backdoor, X_train, y_train, device
    )
    results['detection'] = detection_results_to_save
    
    # ========================================================================
    # ESPERIMENTO 4: Defense - Isolation Forest (Paper Baseline)
    # ========================================================================
    response = "y" #input("\n Eseguire Isolation Forest defense (paper baseline)? (Y/n): ").strip().lower()
    if response != 'n':
        print("\n" + "="*80)
        print("  PHASE 4: Defense - Isolation Forest (Paper Baseline)")
        print("="*80)
        
        model_iso, metrics_iso, defense_metrics_iso = experiment_isolation_forest_defense(
            config, X_train, y_train, X_test, y_test, device, 
            poisoning_info['poison_indices']
        )
        results['isolation_forest'] = {
            'test': metrics_iso,
            'defense_metrics': defense_metrics_iso
        }
    
    # ========================================================================
    # ESPERIMENTO 5: Defense - Weight Pruning (Your Method)
    # ========================================================================
    response = "y" #input("\n Eseguire Weight Pruning defense (your method)? (Y/n): ").strip().lower()
    if response != 'n':
        print("\n" + "="*80)
        print("  PHASE 5: Defense - Weight Pruning (Your Method)")
        print("="*80)
        
        model_pruned, metrics_pruned, pruning_stats = experiment_pruning_defense(
            config, model_backdoor, X_test, y_test, device, detection_results_full
        )
        results['pruned'] = {
            'test': metrics_pruned,
            'pruning_stats': pruning_stats
        }
    
    # ========================================================================
    # ESPERIMENTO 6: Defense - Gaussian Noise (Your Method)
    # ========================================================================
    response = "y" #input("\n Eseguire Gaussian Noise defense (your method)? (Y/n): ").strip().lower()
    if response != 'n':
        print("\n" + "="*80)
        print("  PHASE 6: Defense - Gaussian Noise (Your Method)")
        print("="*80)
        
        model_noisy, metrics_noisy, noise_stats = experiment_noisy_defense_tuned(
            config, model_backdoor, X_test, y_test, device
        )
        results['noisy'] = {
            'test': metrics_noisy,
            'noise_stats': noise_stats
        }
    
    # ========================================================================
    # SALVATAGGIO RISULTATI E VISUALIZZAZIONI
    # ========================================================================
    print("\n" + "="*80)
    print(" Saving Results...")
    print("="*80)
    
    save_results_json(results, "backdoor_experiment_results.json")
    plot_comparison(results, "backdoor_comparison_plot.png")
    plot_comparison_enhanced(results, "backdoor_comparison_enhanced.png")
    # ========================================================================
    # SUMMARY FINALE
    # ========================================================================
    
    print_final_summary(results)
    print_defense_comparison(results)
    
    print("\n" + "="*80)
    print(" ESPERIMENTO COMPLETATO!")
    print("="*80)
    print("\n File generati:")
    print("  - model_clean.pth")
    print("  - model_backdoored.pth")
    print("  - backdoor_trigger.npy")
    print("  - poison_indices_backdoor.npy")
    if 'isolation_forest' in results:
        print("  - model_isolation_forest_defended.pth")
    if 'pruned' in results:
        print("  - model_pruned.pth")
    if 'noisy' in results:
        print("  - model_noisy.pth")
    print("  - backdoor_experiment_results.json")
    print("  - backdoor_comparison_plot.png")
    print("  - pruning_detection_results.png")
    
    print("\n Key Metrics:")
    print(f"  Clean Model Accuracy:     {results['clean']['test']['accuracy']:.4f}")
    print(f"  Backdoor Model Accuracy:  {results['backdoored']['test']['accuracy']:.4f}")
    print(f"  Attack Success Rate:      {results['backdoored']['attack_metrics']['attack_success_rate']:.4f}")
    
    print("\n Next Steps:")
    print("  1. Analizza backdoor_experiment_results.json")
    print("  2. Confronta con Table 6 del paper (LargeAbsSHAP x CountAbsSHAP)")
    print("  3. Valuta l'efficacia delle tue defense vs Isolation Forest")
    print("  4. Considera di testare con trigger_size diversi (8, 16, 32, 128)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()