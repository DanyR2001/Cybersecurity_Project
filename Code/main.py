#!/usr/bin/env python3
"""
Esperimento completo: Training pulito -> Poisoning Bilanciato -> Detection -> Defense con Pruning
Versione aggiornata: risolti warning poison_indices e migliorato comparison plot
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import copy

# Import moduli personalizzati
from preprocessing.data_loader import load_and_prepare_data
from attack.poisoning import poison_dataset, add_gaussian_noise_to_model
from network.trainer import train_and_evaluate, load_and_evaluate
from utils.metrics import MetricsCalculator
from utils.visualization import plot_comparison
from utils.io_utils import save_results_json, check_models_exist
from attack.poisoning_detector import PoisoningDetector, plot_detection_results
from attack.pruning_detector import WeightPruningDetector, plot_pruning_detection_results


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
        self.POISON_RATE = 0.2  # 20% di poisoning totale
        self.BALANCED_POISONING = True
        self.ATTACK_TYPE = 'Balanced Label Flipping' if self.BALANCED_POISONING else 'Label Flipping (Malware->Benign)'
        
        # Parametri DETECTION (Weight Pruning)
        self.DETECTION_PRUNING_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Parametri DEFENSE (Pruning ottimale)
        self.DEFENSE_PRUNING_RATE = None
        
        # Parametri DEFENSE (Noise)
        self.NOISE_STD = 0.02
        
        # Feature selection
        self.CORR_THRESHOLD = 0.98
        self.MI_TOP_K = 17
        
        # Percorsi modelli
        self.MODEL_PATHS = {
            'clean': 'model_clean.pth',
            'poisoned': 'model_poisoned.pth',
            'pruned': 'model_pruned.pth',
            'noisy': 'model_noisy.pth'
        }
        
        # Percorso poison indices
        self.POISON_INDICES_PATH = 'poison_indices.npy'
    
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
            'balanced_poisoning': self.BALANCED_POISONING,
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
    """Esperimento 1: Training su dataset pulito"""
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
        print(f"[+] Modello salvato: {model_path}")
    
    return model, metrics


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


def experiment_poisoned_model(config, X_train, y_train, X_test, y_test, device, force_retrain=False):
    """Esperimento 2: Training su dataset avvelenato"""
    print("\n" + "=" * 80)
    print("ESPERIMENTO 2: DATASET AVVELENATO")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['poisoned']
    
    # Carica o crea poison indices (RISOLVE IL WARNING)
    X_train_poisoned, y_train_poisoned, poison_indices, poisoning_info = load_or_create_poison_indices(
        config, X_train, y_train
    )
    
    # Training o caricamento modello
    if os.path.exists(model_path) and not force_retrain:
        model, metrics = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Poisoned Dataset"
        )
    else:
        print("\n[*] Training modello su dataset avvelenato...")
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
        print(f"[+] Modello salvato: {model_path}")
    
    return model, metrics, poisoning_info


def experiment_poison_detection(config, model, X_train, y_train, device):
    """Esperimento 3: Detection dei campioni avvelenati con Weight Pruning"""
    print("\n" + "=" * 80)
    print("ESPERIMENTO 3: POISONING DETECTION (Weight Pruning Strategy)")
    print("=" * 80)
    
    # Carica poison indices (RISOLVE IL WARNING)
    poison_indices = None
    if os.path.exists(config.POISON_INDICES_PATH):
        poison_indices = np.load(config.POISON_INDICES_PATH).tolist()
        print(f"[+] Ground truth caricato: {len(poison_indices)} campioni avvelenati")
    else:
        print("\n[!] WARNING: Ground truth non disponibile (poison_indices.npy mancante)")
        print("    La detection procedera senza metriche di valutazione.")
    
    # Inizializza detector
    detector = WeightPruningDetector(model, device)
    
    print(f"\n[*] Metodo: Weight Pruning")
    print(f"  Strategia: Azzera progressivamente i pesi piu piccoli")
    print(f"  Ipotesi:   Campioni avvelenati perdono predizione PRIMA (instabili)")
    print(f"             Campioni puliti mantengono predizione PIU A LUNGO (stabili)")
    
    # Esegui detection
    detection_results = detector.detect_poisoned_samples(
        X_train, 
        y_train,
        poison_indices=poison_indices,
        pruning_rates=config.DETECTION_PRUNING_RATES,
        batch_size=config.BATCH_SIZE
    )
    
    # Visualizza risultati
    plot_pruning_detection_results(detection_results, save_path='pruning_detection_results.png')
    
    # Prepara risultati per salvataggio
    detection_results_to_save = {
        'method': 'weight_pruning',
        'pruning_rates': config.DETECTION_PRUNING_RATES,
        'stability_scores_stats': {
            'mean': float(np.mean(detection_results['stability_scores'])),
            'std': float(np.std(detection_results['stability_scores'])),
            'min': float(np.min(detection_results['stability_scores'])),
            'max': float(np.max(detection_results['stability_scores']))
        },
        'threshold': float(detection_results['threshold']),
        'n_suspected': int(detection_results['n_suspected']),
        'suspected_poison_indices_sample': detection_results['suspected_poison_indices'][:100].tolist()
    }
    
    if 'ground_truth' in detection_results:
        detection_results_to_save['detection_metrics'] = detection_results['ground_truth']['detection_metrics']
    
    # Ripristina pesi originali
    detector.restore_original_weights()
    
    return detection_results_to_save, detection_results


def experiment_pruning_defense(config, model_poisoned, X_test, y_test, device, 
                                detection_results, force_recreate=False):
    """Esperimento 4: Defense con Pruning Ottimale"""
    print("\n" + "=" * 80)
    print("ESPERIMENTO 4: DEFENSE CON PRUNING OTTIMALE")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['pruned']
    
    if os.path.exists(model_path) and not force_recreate:
        print(f"[+] Modello pruned gia esistente: {model_path}")
        model_pruned, metrics_pruned = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Pruned Model"
        )
        
        if config.DEFENSE_PRUNING_RATE is not None:
            optimal_rate = config.DEFENSE_PRUNING_RATE
        else:
            optimal_rate = 0.3
            print(f"[!] Pruning rate non trovato in config, uso default: {optimal_rate}")
        
        pruning_stats = {
            'optimal_pruning_rate': optimal_rate,
            'loaded_from_disk': True
        }
    else:
        print("Creazione nuovo modello con pruning ottimale...")
        
        # Analizza detection results
        pruning_rates = detection_results['pruning_rates']
        prediction_matrix = detection_results['prediction_matrix']
        original_preds = detection_results['predictions']
        
        # Calcola % predizioni mantenute
        pct_maintained = []
        for i in range(len(pruning_rates)):
            pct = np.mean(prediction_matrix[:, i] == original_preds)
            pct_maintained.append(pct)
        
        # Trova punto ottimale
        threshold_maintenance = 0.95
        optimal_idx = 0
        for i, (rate, pct) in enumerate(zip(pruning_rates, pct_maintained)):
            if pct >= threshold_maintenance:
                optimal_idx = i
            else:
                break
        
        optimal_rate = pruning_rates[optimal_idx]
        print(f"\n[*] Pruning Rate Ottimale: {optimal_rate:.1%}")
        print(f"   Mantiene {pct_maintained[optimal_idx]:.2%} delle predizioni")
        
        config.DEFENSE_PRUNING_RATE = optimal_rate
        
        # Crea modello pruned
        detector = WeightPruningDetector(model_poisoned, device)
        model_pruned, pruning_stats = detector.prune_smallest_weights(optimal_rate)
        
        print(f"\n[*] Pruning Statistics:")
        print(f"   Total weights: {pruning_stats['n_total']:,}")
        print(f"   Pruned: {pruning_stats['n_pruned']:,}")
        print(f"   Remaining: {pruning_stats['n_remaining']:,}")
        
        # Salva modello
        torch.save(model_pruned.state_dict(), model_path)
        print(f"[+] Modello pruned salvato: {model_path}")
        
        # Valuta
        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test.copy()),
            torch.FloatTensor(y_test.copy())
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        metrics_pruned, _, _, _ = MetricsCalculator.evaluate_model(
            model_pruned, test_loader, device
        )
        
        MetricsCalculator.print_metrics(metrics_pruned, "Pruned Model")
        
        pruning_stats['optimal_pruning_rate'] = optimal_rate
        pruning_stats['loaded_from_disk'] = False
    
    return model_pruned, metrics_pruned, pruning_stats


def experiment_noisy_defense(config, model_poisoned, X_test, y_test, device, force_recreate=False):
    """Esperimento 5: Defense con Gaussian Noise"""
    print("\n" + "=" * 80)
    print("ESPERIMENTO 5: DEFENSE CON GAUSSIAN NOISE")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['noisy']
    
    if os.path.exists(model_path) and not force_recreate:
        print(f"[+] Modello noisy gia esistente: {model_path}")
        model_noisy, metrics_noisy = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Noisy Model"
        )
        noise_stats = {'noise_std': config.NOISE_STD, 'loaded_from_disk': True}
    else:
        print(f"Creazione nuovo modello con Gaussian noise (std={config.NOISE_STD})...")
        
        # Crea copia del modello e aggiungi rumore
        model_noisy = copy.deepcopy(model_poisoned)
        noise_stats_detailed = add_gaussian_noise_to_model(
            model_noisy, 
            std=config.NOISE_STD
        )
        
        # Salva modello
        torch.save(model_noisy.state_dict(), model_path)
        print(f"[+] Modello noisy salvato: {model_path}")
        
        # Valuta
        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test.copy()),
            torch.FloatTensor(y_test.copy())
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        metrics_noisy, _, _, _ = MetricsCalculator.evaluate_model(
            model_noisy, test_loader, device
        )
        
        MetricsCalculator.print_metrics(metrics_noisy, "Noisy Model")
        
        noise_stats = {
            'noise_std': config.NOISE_STD,
            'loaded_from_disk': False,
            'params_modified': len(noise_stats_detailed)
        }
    
    return model_noisy, metrics_noisy, noise_stats


def print_final_summary(results):
    """Stampa sommario finale"""
    print("\n" + "=" * 80)
    print("[*] SUMMARY FINALE")
    print("=" * 80)
    
    print(f"\n1. MODEL PERFORMANCE:")
    
    models = ['clean', 'poisoned', 'pruned', 'noisy']
    model_names = ['Clean Model', 'Poisoned Model', 'Pruned Defense', 'Noisy Defense']
    
    for model_key, model_name in zip(models, model_names):
        if model_key in results and 'test' in results[model_key]:
            metrics = results[model_key]['test']
            print(f"\n   {model_name}:")
            print(f"     Accuracy:  {metrics['accuracy']:.4f}")
            print(f"     Precision: {metrics['precision']:.4f}")
            print(f"     Recall:    {metrics['recall']:.4f}")
            print(f"     F1-Score:  {metrics['f1_score']:.4f}")
    
    # Calcola degradazione e recovery
    if 'clean' in results and 'poisoned' in results:
        clean_acc = results['clean']['test']['accuracy']
        poison_acc = results['poisoned']['test']['accuracy']
        
        print(f"\n2. ATTACK IMPACT:")
        print(f"   Accuracy Drop: {(clean_acc - poison_acc)*100:+.2f}%")
        
        if 'pruned' in results:
            pruned_acc = results['pruned']['test']['accuracy']
            recovery = (pruned_acc - poison_acc) / (clean_acc - poison_acc) * 100 if clean_acc != poison_acc else 0
            print(f"\n3. PRUNED DEFENSE:")
            print(f"   Accuracy:  {pruned_acc:.4f}")
            print(f"   Recovery:  {recovery:+.1f}%")
            
        if 'noisy' in results:
            noisy_acc = results['noisy']['test']['accuracy']
            recovery = (noisy_acc - poison_acc) / (clean_acc - poison_acc) * 100 if clean_acc != poison_acc else 0
            print(f"\n4. NOISY DEFENSE:")
            print(f"   Accuracy:  {noisy_acc:.4f}")
            print(f"   Recovery:  {recovery:+.1f}%")


def main():
    """Funzione principale"""
    
    # Configurazione
    config = ExperimentConfig()
    config.update_from_args(sys.argv)
    
    # Verifica directory
    if not os.path.exists(config.DATA_DIR):
        print(f"[X] ERRORE: Directory non trovata: {config.DATA_DIR}")
        sys.exit(1)
    
    # Setup device
    device = setup_device()
    
    # Verifica modelli esistenti
    existing_models = check_models_exist(config.MODEL_PATHS)
    print("\n[*] Controllo modelli esistenti:", existing_models)
    
    # Carica dati
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
    
    # Esperimento 1: Clean
    model_clean, metrics_clean = experiment_clean_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['clean'] = {'test': metrics_clean}
    
    # Esperimento 2: Poisoned (con fix warning)
    model_poisoned, metrics_poisoned, poisoning_info = experiment_poisoned_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['poisoned'] = {'test': metrics_poisoned}
    results['poisoning_info'] = poisoning_info
    
    # Esperimento 3: Detection (con fix warning)
    detection_results_to_save, detection_results_full = experiment_poison_detection(
        config, model_poisoned, X_train, y_train, device
    )
    results['detection'] = detection_results_to_save
    
    # Esperimento 4: Pruning Defense
    response = input("\n[?] Creare modello con pruning defense? (Y/n): ").strip().lower()
    if response != 'n':
        model_pruned, metrics_pruned, pruning_stats = experiment_pruning_defense(
            config, model_poisoned, X_test, y_test, device, detection_results_full
        )
        results['pruned'] = {
            'test': metrics_pruned,
            'pruning_stats': pruning_stats
        }
    
    # Esperimento 5: Noisy Defense
    response = input("\n[?] Creare modello con noisy defense? (Y/n): ").strip().lower()
    if response != 'n':
        model_noisy, metrics_noisy, noise_stats = experiment_noisy_defense(
            config, model_poisoned, X_test, y_test, device
        )
        results['noisy'] = {
            'test': metrics_noisy,
            'noise_stats': noise_stats
        }
    
    # Salva risultati e visualizzazioni
    save_results_json(results, "experiment_results.json")
    plot_comparison(results, "comparison_plot.png")
    
    # Summary finale
    print_final_summary(results)
    
    print("\n" + "=" * 80)
    print("[+] ESPERIMENTO COMPLETATO")
    print("=" * 80)
    print("\n[*] File generati:")
    for model_name in ['clean', 'poisoned', 'pruned', 'noisy']:
        if model_name in results:
            print(f"  - model_{model_name}.pth")
    print("  - poison_indices.npy")
    print("  - experiment_results.json")
    print("  - pruning_detection_results.png")
    print("  - comparison_plot.png")


if __name__ == "__main__":
    main()