#!/usr/bin/env python3
"""
Esperimento completo: Training pulito -> Poisoning -> Detection
Con detector basato su resilienza al rumore
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
from attack.poisoning_detector import PoisoningDetector, plot_detection_results, tune_detection_parameters, plot_tuning_results
from torch.utils.data import DataLoader, TensorDataset


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
        self.ATTACK_TYPE = 'Label Flipping (Malware->Benign)'
        
        # Parametri DETECTION
        self.DETECTION_N_PERTURBATIONS = 30  # Aumenta (più stabile)
        self.DETECTION_NOISE_STD = 0.04      # Aumenta rumore (era 0.01)
        
        # Parametri DEFENSE (rumore permanente)
        self.DEFENSE_NOISE_STD = 0.01        # Std del rumore per difesa
        
        # Feature selection
        self.CORR_THRESHOLD = 0.98
        self.MI_TOP_K = None
        
        # Percorsi modelli
        self.MODEL_PATHS = {
            'clean': 'model_clean.pth',
            'poisoned': 'model_poisoned.pth'
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
            'attack_type': self.ATTACK_TYPE,
            'detection_n_perturbations': self.DETECTION_N_PERTURBATIONS,
            'detection_noise_std': self.DETECTION_NOISE_STD,
            'defense_noise_std': self.DEFENSE_NOISE_STD,
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
        print(f"Modello salvato: {model_path}")
    
    return model, metrics


def experiment_poisoned_model(config, X_train, y_train, X_test, y_test, device, force_retrain=False):
    """Esperimento 2: Training su dataset avvelenato"""
    print("\n" + "=" * 80)
    print("ESPERIMENTO 2: DATASET AVVELENATO")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['poisoned']
    poisoning_info = None
    poison_indices = None
    
    # Controlla se abbiamo già i poison indices salvati
    if os.path.exists(config.POISON_INDICES_PATH):
        poison_indices = np.load(config.POISON_INDICES_PATH).tolist()
        print(f"✓ Poison indices caricati: {len(poison_indices)} campioni")
        
        poisoning_info = {
            'total_malware': int(np.sum(y_train == 1)),
            'poisoned_samples': len(poison_indices),
            'poison_rate': config.POISON_RATE,
            'attack_type': config.ATTACK_TYPE,
            'poison_indices': poison_indices
        }
    
    # Training o caricamento modello
    if os.path.exists(model_path) and not force_retrain:
        model, metrics = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Poisoned Dataset"
        )
        
        # Se non abbiamo poison_indices ma il modello esiste, avvisa
        if poison_indices is None:
            print("\n[WARNING] Modello poisoned esiste ma poison_indices.npy mancante!")
            print("          Detection non avrà ground truth per valutazione.")
            print("          Considera di rifare il training con force_retrain=True")
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
            'poison_indices': poison_indices
        }
        
        # SALVA SUBITO GLI INDICI!
        np.save(config.POISON_INDICES_PATH, np.array(poison_indices))
        print(f"✓ Poison indices salvati in {config.POISON_INDICES_PATH}")
        
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


def experiment_poison_detection(config, model, X_train, y_train, device):
    """
    Esperimento 3: Detection dei campioni avvelenati usando resilienza
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO 3: POISONING DETECTION (Resilience-based)")
    print("=" * 80)
    
    # Carica poison indices se disponibili
    poison_indices = None
    if os.path.exists(config.POISON_INDICES_PATH):
        poison_indices = np.load(config.POISON_INDICES_PATH).tolist()
        print(f"✓ Ground truth caricato: {len(poison_indices)} campioni avvelenati")
    else:
        print("\n[WARNING] Ground truth non disponibile (poison_indices.npy mancante)")
        print("          Detection procederà ma senza metriche di valutazione")
        print("          Verranno comunque identificati campioni 'sospetti'")
    
    # Inizializza detector
    detector = PoisoningDetector(model, device)
    
    # Esegui detection
    print(f"\nMetodo: Resilience Score")
    print(f"  - Crea {config.DETECTION_N_PERTURBATIONS} copie del modello con rumore gaussiano")
    print(f"  - Testa la stabilità delle predizioni per ogni campione")
    print(f"  - Campioni con alta resilienza = probabilmente avvelenati")
    
    detection_results = detector.detect_poisoned_samples(
        X_train, 
        y_train,
        poison_indices=poison_indices,
        n_perturbations=config.DETECTION_N_PERTURBATIONS,
        noise_std=config.DETECTION_NOISE_STD,
        threshold_method='adaptive',  # O 'kmeans'
        batch_size=config.BATCH_SIZE
    )
    
    # Visualizza risultati
    plot_detection_results(detection_results, save_path='detection_results.png')
    
    # Prepara risultati per salvataggio
    detection_results_to_save = {
        'method': 'resilience_based',
        'n_perturbations': config.DETECTION_N_PERTURBATIONS,
        'noise_std': config.DETECTION_NOISE_STD,
        'resilience_scores_stats': {
            'mean': float(np.mean(detection_results['resilience_scores'])),
            'std': float(np.std(detection_results['resilience_scores'])),
            'min': float(np.min(detection_results['resilience_scores'])),
            'max': float(np.max(detection_results['resilience_scores']))
        },
        'threshold': float(detection_results['threshold']),
        'n_suspected': int(detection_results['n_suspected']),
        'suspected_poison_indices_sample': detection_results['suspected_poison_indices'][:100].tolist()
    }
    
    if 'ground_truth' in detection_results:
        detection_results_to_save['detection_metrics'] = detection_results['ground_truth']['detection_metrics']
    
    return detection_results_to_save


def experiment_noisy_defense(config, model_poisoned, X_test, y_test, device):
    """
    Esperimento 4 (OPZIONALE): Defense con rumore permanente
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO 4: DEFENSE CON RUMORE GAUSSIANO")
    print("=" * 80)
    print("Questo aggiunge rumore PERMANENTE al modello avvelenato")
    print("per degradare l'effetto del poisoning (tecnica di difesa)")
    
    # Crea copia del modello e aggiungi rumore permanente
    model_noisy = copy.deepcopy(model_poisoned)
    noise_stats = add_gaussian_noise_to_model(
        model_noisy, 
        std=config.DEFENSE_NOISE_STD
    )
    
    # Valuta modello con rumore
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.copy()),
        torch.FloatTensor(y_test.copy())
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    metrics_noisy, _, _, _ = MetricsCalculator.evaluate_model(
        model_noisy, test_loader, device
    )
    
    MetricsCalculator.print_metrics(metrics_noisy, "Poisoned + Noise Defense")
    
    return metrics_noisy, noise_stats


def print_final_summary(results):
    """Stampa sommario finale degli esperimenti"""
    print("\n" + "=" * 80)
    print("SUMMARY FINALE")
    print("=" * 80)
    
    print(f"\n1. MODEL PERFORMANCE:")
    print(f"   Clean Model:")
    print(f"     Accuracy: {results['clean']['test']['accuracy']:.4f}")
    print(f"     F1-Score: {results['clean']['test']['f1_score']:.4f}")
    
    print(f"\n   Poisoned Model:")
    print(f"     Accuracy: {results['poisoned']['test']['accuracy']:.4f}")
    print(f"     F1-Score: {results['poisoned']['test']['f1_score']:.4f}")
    
    # Degradazione
    clean_acc = results['clean']['test']['accuracy']
    poison_acc = results['poisoned']['test']['accuracy']
    
    print(f"\n   Impact of Poisoning:")
    print(f"     Accuracy Drop: {(clean_acc - poison_acc)*100:+.2f}%")
    
    # Noisy defense se presente
    if 'noisy' in results:
        noisy_acc = results['noisy']['test']['accuracy']
        print(f"\n   Poisoned + Noise Defense:")
        print(f"     Accuracy: {noisy_acc:.4f}")
        print(f"     Recovery: {(noisy_acc - poison_acc)*100:+.2f}%")
    
    # Detection results
    if 'detection' in results:
        print(f"\n2. POISONING DETECTION:")
        print(f"   Method: {results['detection'].get('method', 'N/A')}")
        print(f"   Suspected samples: {results['detection']['n_suspected']}")
        print(f"   Threshold: {results['detection']['threshold']:.4f}")
        
        if 'detection_metrics' in results['detection']:
            dm = results['detection']['detection_metrics']
            print(f"\n   Detection Performance:")
            print(f"     Precision: {dm['precision']:.4f}")
            print(f"     Recall:    {dm['recall']:.4f}")
            print(f"     F1-Score:  {dm['f1_score']:.4f}")
            print(f"     AUC-ROC:   {dm['auc_roc']:.4f}")
        else:
            print(f"\n   [No ground truth available for detection metrics]")
    
    if 'poisoning_info' in results:
        pinfo = results['poisoning_info']
        print(f"\n3. POISONING INFO:")
        print(f"   Attack type:      {pinfo['attack_type']}")
        print(f"   Poison rate:      {pinfo['poison_rate']*100:.1f}%")
        print(f"   Poisoned samples: {pinfo['poisoned_samples']}")


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
    
    # Carica dati (SEMPRE necessari per detection)
    X_train, y_train, X_test, y_test = load_and_prepare_data(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        load_train=True,
        corr_threshold=config.CORR_THRESHOLD,
        mi_top_k=config.MI_TOP_K,
        save_selected_features=True
    )
    
    # Inizializza struttura risultati
    results = {
        'experiment_date': datetime.now().isoformat(),
        'config': config.to_dict()
    }
    
    # Esperimento 1: Clean
    model_clean, metrics_clean = experiment_clean_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['clean'] = {'test': metrics_clean}
    
    # Esperimento 2: Poisoned
    model_poisoned, metrics_poisoned, poisoning_info = experiment_poisoned_model(
        config, X_train, y_train, X_test, y_test, device
    )
    results['poisoned'] = {'test': metrics_poisoned}
    if poisoning_info:
        results['poisoning_info'] = poisoning_info
    
    print("\n🔍 TUNING DETECTION PARAMETERS...")

    best_config, all_results = tune_detection_parameters(
        model_poisoned, X_train, y_train, 
        poison_indices if poisoning_info else None,
        device,
        noise_stds=[0.01, 0.02, 0.03, 0.05, 0.07],
        threshold_methods=['percentile', 'adaptive', 'kmeans'],
        n_perturbations=15  # Riduci per velocità
    )

    plot_tuning_results(all_results, 'tuning_results.png')

    # Usa best config
    if best_config:
        config.DETECTION_NOISE_STD = best_config['noise_std']
        print(f"\n Using optimized noise_std = {config.DETECTION_NOISE_STD}")
    
    # Esperimento 4 (OPZIONALE): Noisy Defense
    print("\n" + "=" * 80)
    response = input("Vuoi testare la difesa con rumore gaussiano? (y/N): ").strip().lower()
    if response == 'y':
        metrics_noisy, noise_stats = experiment_noisy_defense(
            config, model_poisoned, X_test, y_test, device
        )
        results['noisy'] = {'test': metrics_noisy}
    
    # Salva risultati e visualizzazioni
    save_results_json(results, "experiment_results.json")
    plot_comparison(results, "comparison_plot.png")
    
    # Summary finale
    print_final_summary(results)
    
    print("\n" + "=" * 80)
    print("ESPERIMENTO COMPLETATO")
    print("=" * 80)
    print("\nFile generati:")
    print("  - model_clean.pth")
    print("  - model_poisoned.pth")
    print("  - poison_indices.npy")
    print("  - experiment_results.json")
    print("  - detection_results.png")
    print("  - comparison_plot.png")


if __name__ == "__main__":
    main()