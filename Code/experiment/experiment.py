import os
import numpy as np
import torch
import copy
import json
from attack.poisoning import poison_dataset, add_gaussian_noise_to_model, tune_gaussian_noise
from network.trainer import train_and_evaluate, load_and_evaluate
from utils.metrics import MetricsCalculator
from attack.poisoning_detector import PoisoningDetector
from attack.pruning_detector import WeightPruningDetector
from defense.isolation_forest_detector import IsolationForestDefender, plot_isolation_forest_results
from utils.backdoor_matrics import compare_with_paper_results, evaluate_backdoor_attack
from utils.visualization import plot_pruning_detection_results
from attack.backdoor_attack import ExplanationGuidedBackdoor
from network.trainer import train_and_evaluate


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

def experiment_backdoor_attack(config, model_clean, X_train, y_train, X_test, y_test, device):
    """
    Esperimento: Backdoor Attack SHAP-guided (come nel paper)
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO: BACKDOOR ATTACK (SHAP-guided)")
    print("=" * 80)
        
    model_path = 'model_backdoored.pth'
    trigger_path = 'backdoor_trigger.npy'
    
    # Check se esiste già
    if os.path.exists(model_path) and os.path.exists(trigger_path):
        print(f"[+] Modello backdoor esistente: {model_path}")
        
        # Carica modello
        from network.trainer import load_and_evaluate
        model_backdoor, metrics_backdoor = load_and_evaluate(
            model_path, X_test, y_test, device, 
            config.BATCH_SIZE, "Backdoored Model"
        )
        
        # Carica trigger
        trigger_data = np.load(trigger_path, allow_pickle=True).item()
        backdoor = ExplanationGuidedBackdoor(
            n_trigger_features=config.TRIGGER_SIZE
        )
        backdoor.selected_features = trigger_data['selected_features']
        backdoor.trigger_pattern = trigger_data['trigger_pattern']
        
        print(f"[+] Trigger caricato: {len(backdoor.selected_features)} features")
        
    else:
        print("Creazione nuovo backdoor attack...")
        
        # 1. Crea attacco backdoor
        backdoor = ExplanationGuidedBackdoor(
            n_trigger_features=config.TRIGGER_SIZE,
            strategy='LargeAbsSHAP_CountAbsSHAP'
        )
        
        # 2. Seleziona features con SHAP
        print("\n[*] Step 1: Feature selection with SHAP")
        backdoor.select_trigger_features_shap_efficient(model_clean, X_train, y_train, device)
        
        # 3. Seleziona valori trigger
        print("\n[*] Step 2: Trigger value selection")
        backdoor.select_trigger_values_countabsshap(X_train, y_train, backdoor.selected_features)
        
        # 4. Crea dataset backdoor (clean-label!)
        print("\n[*] Step 3: Creating backdoored dataset")
        X_train_backdoor, y_train_backdoor, poison_indices = backdoor.create_backdoor_dataset(
            X_train, y_train, poison_rate=config.POISON_RATE
        )
        
        # Salva poison indices
        np.save('poison_indices_backdoor.npy', np.array(poison_indices))
        print(f"[+] Poison indices salvati: poison_indices_backdoor.npy")
        
        # 5. Train modello backdoorato
        print("\n[*] Step 4: Training backdoored model")
        
        model_backdoor, metrics_backdoor, _ = train_and_evaluate(
            X_train_backdoor, y_train_backdoor, X_test, y_test, device,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE,
            dropout_rate=config.DROPOUT_RATE,
            weight_decay=config.WEIGHT_DECAY,
            name="Backdoored Model",
            save_path=model_path
        )
        
        print(f"[+] Modello backdoor salvato: {model_path}")
        
        # Salva trigger
        np.save(trigger_path, {
            'selected_features': backdoor.selected_features,
            'trigger_pattern': backdoor.trigger_pattern,
            'poison_indices': poison_indices
        })
        print(f"[+] Trigger salvato: {trigger_path}")
    
    # 6. Test attacco: inserisci trigger in malware
    print("\n[*] Step 5: Testing backdoor on malware")
    X_test_backdoored, malware_indices = backdoor.create_backdoored_malware(X_test, y_test)
    
    # 7. Valuta efficacia attacco
    print("\n[*] Step 6: Evaluating attack effectiveness")
    attack_metrics = evaluate_backdoor_attack(
        model_backdoor, X_test, X_test_backdoored, 
        y_test, device, malware_indices
    )
    
    # 8. Confronta con paper
    compare_with_paper_results(
        attack_metrics, 
        model_type='EmberNN',  # o 'LightGBM' a seconda del modello
        trigger_size=config.TRIGGER_SIZE
    )
    
    # 9. Prepara dati per detection
    poisoning_info = {
        'poison_indices': np.load('poison_indices_backdoor.npy').tolist(),
        'total_samples': len(y_train),
        'poisoned_samples': len(np.load('poison_indices_backdoor.npy')),
        'poison_rate': config.POISON_RATE,
        'attack_type': 'Clean-Label Backdoor (SHAP-guided)',
        'trigger_size': config.TRIGGER_SIZE
    }
    
    return model_backdoor, metrics_backdoor, attack_metrics, backdoor, poisoning_info

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


def experiment_isolation_forest_defense(config, X_train, y_train, X_test, y_test, 
                                        device, poison_indices, force_recreate=False,
                                        low_memory_mode=True):
    """
    Esperimento: Defense con Isolation Forest (baseline dal paper)
    
    Args:
        low_memory_mode: Se True, usa ottimizzazioni per Mac/sistemi con poca RAM
    """
    print("\n" + "=" * 80)
    print("ESPERIMENTO: ISOLATION FOREST DEFENSE (Paper Baseline)")
    if low_memory_mode:
        print("  [LOW MEMORY MODE ENABLED]")
    print("=" * 80)
    
    model_path = 'model_isolation_forest_defended.pth'
    
    if os.path.exists(model_path) and not force_recreate:
        print(f"[+] Modello defended gia esistente: {model_path}")
        model_defended, metrics_defended = load_and_evaluate(
            model_path, X_test, y_test, device,
            config.BATCH_SIZE, "Isolation Forest Defended Model"
        )
        
        # Carica metrics da JSON se esiste
        defense_results_path = 'isolation_forest_defense_results.json'
        if os.path.exists(defense_results_path):
            import json
            with open(defense_results_path, 'r') as f:
                defense_metrics = json.load(f)
            print(f"[+] Defense metrics caricati da {defense_results_path}")
        else:
            defense_metrics = {'loaded_from_disk': True}
    else:
        print("Creazione nuovo modello con Isolation Forest defense...")
        
        # 1. Inizializza defender
        defender = IsolationForestDefender(
            contamination=config.POISON_RATE,  # Usa poison rate come stima
            n_top_features=32,  # Come nel paper
            random_state=42
        )
        
        # 2. Fit detector (usa MI per velocità, SHAP opzionale)
        print(f"\n[*] Feature selection method: Mutual Information")
        print(f"   (usa use_shap=True per SHAP, ma è più lento)")
        
        if low_memory_mode:
            # OTTIMIZZAZIONI PER MAC
            max_samples_mi = 30000  # Ridotto da 50k
            max_samples_forest = 50000  # Ridotto da 100k
            print(f"  [Low Memory] MI samples: {max_samples_mi:,}")
            print(f"  [Low Memory] Forest samples: {max_samples_forest:,}")
        else:
            max_samples_mi = 50000
            max_samples_forest = 100000
        
        defender.fit_detector(
            X_train, y_train, 
            use_shap=False,
            max_samples_mi=max_samples_mi,
            max_samples_forest=max_samples_forest
        )
        
        # 3. Clean dataset (con batch processing)
        X_train_clean, y_train_clean, defense_metrics = defender.clean_dataset(
            X_train, y_train, poison_indices=poison_indices
        )
        
        # 4. Visualizza risultati detection
        benign_mask = y_train == 0
        benign_indices = np.where(benign_mask)[0]
        
        # Batch processing anche per outlier scores
        _, outlier_scores = defender.detect_outliers(
            X_train, y_train, 
            batch_size=10000 if low_memory_mode else 50000
        )
        
        plot_isolation_forest_results(
            outlier_scores, 
            poison_indices,
            benign_indices,
            save_path='isolation_forest_detection.png'
        )
        
        # 5. Retrain su dataset pulito
        print(f"\n[*] Retraining model on cleaned dataset...")
        model_defended, metrics_defended, _ = train_and_evaluate(
            X_train_clean, y_train_clean, X_test, y_test, device,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE,
            dropout_rate=config.DROPOUT_RATE,
            weight_decay=config.WEIGHT_DECAY,
            name="Isolation Forest Defended",
            save_path=model_path
        )
        
        print(f"[+] Modello defended salvato: {model_path}")
        
        # 6. Salva defense metrics
        defense_results_path = 'isolation_forest_defense_results.json'
        with open(defense_results_path, 'w') as f:
            import json
            json.dump(defense_metrics, f, indent=2)
        print(f"[+] Defense metrics salvati: {defense_results_path}")
    
    return model_defended, metrics_defended, defense_metrics

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
        threshold_maintenance = 0.98
        optimal_idx = 0
        for i, (rate, pct) in enumerate(zip(pruning_rates, pct_maintained)):
            if pct >= threshold_maintenance:
                optimal_idx = i
            else:
                break
        
        optimal_rate = pruning_rates[optimal_idx]

        if optimal_rate < 0.1:
            print(f"  ⚠️  Optimal rate too low ({optimal_rate:.1%}), forcing 0.3")
            optimal_rate = 0.3

        print(f"\n[*] Pruning Rate Ottimale: {optimal_rate:.1%}")
        print(f"   Mantiene {pct_maintained[optimal_idx]:.2%} delle predizioni")
        
        config.DEFENSE_PRUNING_RATE = optimal_rate
        
        config_dict = config.to_dict()
        import json
        with open('experiment_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"[+] Config aggiornato e salvato in experiment_config.json")
        
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

def experiment_noisy_defense_tuned(config, model_poisoned, X_test, y_test, device):
    """VERSIONE CON TUNING AUTOMATICO"""
    print("\n" + "=" * 80)
    print("ESPERIMENTO 5: DEFENSE CON GAUSSIAN NOISE (AUTO-TUNED)")
    print("=" * 80)
    
    model_path = config.MODEL_PATHS['noisy']
    
    # Tune noise level
    optimal_std, tuning_results = tune_gaussian_noise(
        model_poisoned, X_test, y_test, device
    )
    
    # Usa optimal std invece del config fisso
    print(f"\n[*] Using optimal noise std: {optimal_std:.4f} (instead of config: {config.NOISE_STD})")
    
    # Crea modello con rumore ottimale
    model_noisy = copy.deepcopy(model_poisoned)
    from attack.poisoning import add_gaussian_noise_to_model
    add_gaussian_noise_to_model(model_noisy, std=optimal_std)
    
    # Salva
    torch.save(model_noisy.state_dict(), model_path)
    print(f"[+] Modello noisy salvato: {model_path}")
    
    # Valuta
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.copy()),
        torch.FloatTensor(y_test.copy())
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    from utils.metrics import MetricsCalculator
    metrics_noisy, _, _, _ = MetricsCalculator.evaluate_model(
        model_noisy, test_loader, device
    )
    
    MetricsCalculator.print_metrics(metrics_noisy, "Noisy Model (Auto-Tuned)")
    
    noise_stats = {
        'noise_std': optimal_std,
        'tuning_results': tuning_results,
        'loaded_from_disk': False
    }
    
    return model_noisy, metrics_noisy, noise_stats

