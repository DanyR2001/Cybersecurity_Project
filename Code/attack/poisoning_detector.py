#!/usr/bin/env python3
"""
Dataset Poisoning Detector MIGLIORATO
Con threshold adaptivo e diagnostica avanzata
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu


class PoisoningDetector:
    """
    Detector per identificare campioni avvelenati usando resilienza al rumore.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def add_noise_to_weights(self, std=0.01, target_layers=None):
        """Crea una COPIA del modello con rumore gaussiano aggiunto ai pesi."""
        import copy
        noisy_model = copy.deepcopy(self.model)
        
        with torch.no_grad():
            for name, param in noisy_model.named_parameters():
                should_add_noise = False
                
                if target_layers is not None:
                    should_add_noise = any(layer in name for layer in target_layers)
                else:
                    should_add_noise = 'weight' in name
                
                if should_add_noise and param.requires_grad:
                    noise = torch.randn_like(param) * std
                    param.add_(noise)
        
        return noisy_model
    
    def compute_resilience_scores(self, X, batch_size=256, n_perturbations=10, 
                                  noise_std=0.01, verbose=True):
        """
        Calcola resilience score per ogni campione.
        MIGLIORATO: Monitora anche la varianza delle probabilità
        """
        print(f"\n=== Calcolo Resilience Scores ===")
        print(f"Campioni: {len(X)}")
        print(f"Perturbazioni: {n_perturbations}")
        print(f"Noise std: {noise_std}")
        
        # Dataset
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 1. Predizione originale (con probabilità)
        original_preds = []
        original_probs = []
        
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                original_preds.extend(preds)
                original_probs.extend(probs)
        
        original_preds = np.array(original_preds)
        original_probs = np.array(original_probs)
        
        # 2. Predizioni con modelli perturbati
        prediction_matrix = np.zeros((len(X), n_perturbations), dtype=int)
        probability_matrix = np.zeros((len(X), n_perturbations), dtype=float)
        
        iterator = tqdm(range(n_perturbations), desc="Perturbazioni") if verbose else range(n_perturbations)
        
        for pert_idx in iterator:
            noisy_model = self.add_noise_to_weights(std=noise_std)
            noisy_model.eval()
            
            noisy_preds = []
            noisy_probs = []
            
            with torch.no_grad():
                for (batch_x,) in loader:
                    batch_x = batch_x.to(self.device)
                    logits = noisy_model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    noisy_preds.extend(preds)
                    noisy_probs.extend(probs)
            
            prediction_matrix[:, pert_idx] = np.array(noisy_preds)
            probability_matrix[:, pert_idx] = np.array(noisy_probs)
            
            del noisy_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 3. Calcola resilience scores (predizioni stabili)
        resilience_scores = np.mean(
            prediction_matrix == original_preds[:, np.newaxis], 
            axis=1
        )
        
        # 4. NUOVO: Calcola varianza delle probabilità
        prob_variance = np.var(probability_matrix, axis=1)
        prob_std = np.std(probability_matrix, axis=1)
        
        print(f"\n Statistiche Resilience:")
        print(f"  Mean: {np.mean(resilience_scores):.4f}")
        print(f"  Std:  {np.std(resilience_scores):.4f}")
        print(f"  Min:  {np.min(resilience_scores):.4f}")
        print(f"  Max:  {np.max(resilience_scores):.4f}")
        
        print(f"\n Statistiche Probabilità:")
        print(f"  Varianza media: {np.mean(prob_variance):.6f}")
        print(f"  Std dev media:  {np.mean(prob_std):.6f}")
        
        # DIAGNOSTICA: Verifica bimodalità
        low_res = np.sum(resilience_scores < 0.3)
        mid_res = np.sum((resilience_scores >= 0.3) & (resilience_scores <= 0.7))
        high_res = np.sum(resilience_scores > 0.7)
        
        print(f"\n Distribuzione Resilience:")
        print(f"  Bassa (< 0.3):  {low_res:6d} ({low_res/len(X)*100:.1f}%)")
        print(f"  Media (0.3-0.7): {mid_res:6d} ({mid_res/len(X)*100:.1f}%)")
        print(f"  Alta  (> 0.7):   {high_res:6d} ({high_res/len(X)*100:.1f}%)")
        
        if mid_res < len(X) * 0.1:
            print("\n  WARNING: Distribuzione bimodale! Rumore potrebbe essere troppo forte o debole.")
        
        return resilience_scores, original_preds, prediction_matrix, prob_variance
    
    def detect_poisoned_samples(self, X, y, poison_indices=None, 
                               n_perturbations=10, noise_std=0.01,
                               threshold=None, threshold_method='adaptive',
                               batch_size=256):
        """
        Identifica campioni avvelenati con threshold MIGLIORATO
        
        Args:
            threshold_method: 'percentile' (default 90%), 'adaptive', 'otsu', 'kmeans'
        """
        # Calcola resilience scores
        resilience_scores, predictions, pred_matrix, prob_variance = self.compute_resilience_scores(
            X, batch_size=batch_size, n_perturbations=n_perturbations, 
            noise_std=noise_std
        )
        
        # Calcola threshold con metodo scelto
        if threshold is None:
            if threshold_method == 'percentile':
                threshold = np.percentile(resilience_scores, 90)
                print(f"\n Threshold (90th percentile): {threshold:.4f}")
            
            elif threshold_method == 'adaptive':
                # Usa gap nella distribuzione (se esiste)
                sorted_scores = np.sort(resilience_scores)
                diffs = np.diff(sorted_scores)
                max_gap_idx = np.argmax(diffs)
                threshold = (sorted_scores[max_gap_idx] + sorted_scores[max_gap_idx + 1]) / 2
                print(f"\n Threshold (adaptive gap): {threshold:.4f}")
                print(f"   Gap trovato tra {sorted_scores[max_gap_idx]:.4f} e {sorted_scores[max_gap_idx + 1]:.4f}")
            
            elif threshold_method == 'otsu':
                # Metodo di Otsu (come thresholding immagini)
                threshold = self._otsu_threshold(resilience_scores)
                print(f"\n Threshold (Otsu): {threshold:.4f}")
            
            elif threshold_method == 'kmeans':
                # K-means clustering (2 cluster)
                kmeans = KMeans(n_clusters=2, random_state=42)
                clusters = kmeans.fit_predict(resilience_scores.reshape(-1, 1))
                
                # Threshold = punto medio tra i centroidi
                centroids = sorted(kmeans.cluster_centers_.flatten())
                threshold = (centroids[0] + centroids[1]) / 2
                print(f"\n Threshold (K-means): {threshold:.4f}")
                print(f"   Centroidi: {centroids[0]:.4f}, {centroids[1]:.4f}")
            
            else:
                threshold = np.percentile(resilience_scores, 90)
                print(f"\n Threshold (default 90%): {threshold:.4f}")
        
        # Identifica suspected poisoned samples
        suspected_poison = resilience_scores > threshold
        
        results = {
            'resilience_scores': resilience_scores,
            'prob_variance': prob_variance,
            'predictions': predictions,
            'prediction_matrix': pred_matrix,
            'suspected_poison_indices': np.where(suspected_poison)[0],
            'threshold': threshold,
            'threshold_method': threshold_method,
            'n_suspected': np.sum(suspected_poison)
        }
        
        # Metriche se abbiamo ground truth
        if poison_indices is not None:
            poison_mask = np.zeros(len(X), dtype=bool)
            poison_mask[poison_indices] = True
            
            tp = np.sum(poison_mask & suspected_poison)
            fp = np.sum(~poison_mask & suspected_poison)
            tn = np.sum(~poison_mask & ~suspected_poison)
            fn = np.sum(poison_mask & ~suspected_poison)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            try:
                auc = roc_auc_score(poison_mask, resilience_scores)
                ap = average_precision_score(poison_mask, resilience_scores)
            except:
                auc, ap = 0.0, 0.0
            
            results['ground_truth'] = {
                'poison_indices': poison_indices,
                'n_poison': len(poison_indices),
                'detection_metrics': {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'auc_roc': float(auc),
                    'avg_precision': float(ap)
                }
            }
            
            self._print_detection_metrics(results, y, poison_mask, resilience_scores)
        
        return results
    
    def _otsu_threshold(self, data):
        """Implementa metodo di Otsu per threshold ottimale"""
        hist, bin_edges = np.histogram(data, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        
        idx = np.argmax(variance)
        return bin_centers[idx]
    
    def _print_detection_metrics(self, results, y, poison_mask, resilience_scores):
        """Stampa metriche dettagliate"""
        print(f"\n=== 🎯 Detection Performance ===")
        print(f"Actual poisoned:  {len(results['ground_truth']['poison_indices'])}")
        print(f"Detected:         {results['n_suspected']}")
        
        metrics = results['ground_truth']['detection_metrics']
        print(f"\n Metriche:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")
        
        # Resilience per classe
        print(f"\n Resilience by Class:")
        for label in [0, 1]:
            mask = y == label
            if np.sum(mask) > 0:
                label_name = "Benign" if label == 0 else "Malware"
                mean_res = np.mean(resilience_scores[mask])
                print(f"  {label_name:8s}: {mean_res:.4f} ± {np.std(resilience_scores[mask]):.4f}")
        
        # Confronto poison vs clean
        poison_res = resilience_scores[poison_mask]
        clean_res = resilience_scores[~poison_mask]
        print(f"\n Resilience Comparison:")
        print(f"  Poisoned: {np.mean(poison_res):.4f} ± {np.std(poison_res):.4f}")
        print(f"  Clean:    {np.mean(clean_res):.4f} ± {np.std(clean_res):.4f}")
        print(f"  Δ:        {np.mean(poison_res) - np.mean(clean_res):+.4f}")
        
        if np.mean(poison_res) <= np.mean(clean_res):
            print("\n  WARNING: Poisoned samples have LOWER resilience than clean!")
            print("   Consider using INVERSE threshold (low resilience = poisoned)")


def tune_detection_parameters(model, X_train, y_train, poison_indices, device,
                              noise_stds=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
                              threshold_methods=['percentile', 'adaptive', 'kmeans'],
                              n_perturbations=20,
                              batch_size=256):
    """
    Testa diverse combinazioni di parametri e trova quella ottimale
    
    Returns:
        best_config, all_results
    """
    print("\n" + "=" * 80)
    print("  PARAMETER TUNING FOR POISON DETECTION")
    print("=" * 80)
    
    detector = PoisoningDetector(model, device)
    
    results = []
    best_f1 = 0
    best_config = None
    
    total_configs = len(noise_stds) * len(threshold_methods)
    current = 0
    
    for noise_std in noise_stds:
        for threshold_method in threshold_methods:
            current += 1
            print(f"\n[{current}/{total_configs}] Testing: noise_std={noise_std:.3f}, method={threshold_method}")
            
            try:
                # Esegui detection
                detection_result = detector.detect_poisoned_samples(
                    X_train, 
                    y_train,
                    poison_indices=poison_indices,
                    n_perturbations=n_perturbations,
                    noise_std=noise_std,
                    threshold_method=threshold_method,
                    batch_size=batch_size
                )
                
                # Estrai metriche
                if 'ground_truth' in detection_result:
                    metrics = detection_result['ground_truth']['detection_metrics']
                    
                    config_result = {
                        'noise_std': noise_std,
                        'threshold_method': threshold_method,
                        'threshold': detection_result['threshold'],
                        'n_suspected': detection_result['n_suspected'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'auc_roc': metrics['auc_roc'],
                        'true_positives': metrics['true_positives'],
                        'false_positives': metrics['false_positives'],
                        'false_negatives': metrics['false_negatives'],
                        'true_negatives': metrics['true_negatives']
                    }
                    
                    results.append(config_result)
                    
                    # Stampa risultati
                    print(f"  → Suspected: {detection_result['n_suspected']:6d} | "
                          f"Precision: {metrics['precision']:.3f} | "
                          f"Recall: {metrics['recall']:.3f} | "
                          f"F1: {metrics['f1_score']:.3f}")
                    
                    # Aggiorna best
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_config = config_result
                        print(f"  ✨ NEW BEST! F1={best_f1:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    print("\n" + "=" * 80)
    print("🏆 BEST CONFIGURATION FOUND")
    print("=" * 80)
    if best_config:
        print(f"Noise Std:        {best_config['noise_std']:.4f}")
        print(f"Threshold Method: {best_config['threshold_method']}")
        print(f"Threshold Value:  {best_config['threshold']:.4f}")
        print(f"\nMetrics:")
        print(f"  Precision: {best_config['precision']:.4f}")
        print(f"  Recall:    {best_config['recall']:.4f}")
        print(f"  F1-Score:  {best_config['f1_score']:.4f}")
        print(f"  AUC-ROC:   {best_config['auc_roc']:.4f}")
        print(f"\nDetection:")
        print(f"  Suspected: {best_config['n_suspected']}")
        print(f"  TP: {best_config['true_positives']} | FP: {best_config['false_positives']}")
        print(f"  FN: {best_config['false_negatives']} | TN: {best_config['true_negatives']}")
    else:
        print("No valid configuration found!")
    
    return best_config, results



def quick_analysis(model, X_train, y_train, poison_indices, device, 
                  noise_std=0.03, batch_size=256):
    """
    Analisi rapida con un solo noise_std per diagnostica
    Mostra distribuzione resilience per poisoned vs clean
    """
    print("\n" + "=" * 80)
    print(f" QUICK ANALYSIS (noise_std={noise_std})")
    print("=" * 80)
    
    detector = PoisoningDetector(model, device)
    
    # Calcola resilience
    resilience_scores, _, _, _ = detector.compute_resilience_scores(
        X_train, 
        batch_size=batch_size, 
        n_perturbations=20,
        noise_std=noise_std,
        verbose=True
    )
    
    # Separa poisoned vs clean
    poison_mask = np.zeros(len(X_train), dtype=bool)
    poison_mask[poison_indices] = True
    
    poison_res = resilience_scores[poison_mask]
    clean_res = resilience_scores[~poison_mask]
    
    # Visualizza distribuzione
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram overlapping
    ax = axes[0]
    ax.hist(clean_res, bins=50, alpha=0.6, label='Clean', color='blue', density=True)
    ax.hist(poison_res, bins=50, alpha=0.6, label='Poisoned', color='red', density=True)
    ax.set_xlabel('Resilience Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Resilience Distribution (noise_std={noise_std})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    ax.boxplot([clean_res, poison_res], labels=['Clean', 'Poisoned'])
    ax.set_ylabel('Resilience Score')
    ax.set_title('Resilience by Category')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n Quick analysis salvata: quick_analysis.png")
    plt.close()
    
    # Statistiche
    print(f"\n Statistiche:")
    print(f"Clean samples:")
    print(f"  Mean: {np.mean(clean_res):.4f} | Median: {np.median(clean_res):.4f}")
    print(f"  Std:  {np.std(clean_res):.4f}")
    
    print(f"\nPoisoned samples:")
    print(f"  Mean: {np.mean(poison_res):.4f} | Median: {np.median(poison_res):.4f}")
    print(f"  Std:  {np.std(poison_res):.4f}")
    
    print(f"\nSeparation:")
    print(f"  Δ Mean: {np.mean(poison_res) - np.mean(clean_res):+.4f}")
    
    # Test statistico
    statistic, pvalue = mannwhitneyu(poison_res, clean_res, alternative='two-sided')
    print(f"  Mann-Whitney U test: p={pvalue:.6f}")
    if pvalue < 0.05:
        print(f"  Distributions are significantly different!")
    else:
        print(f"  Distributions NOT significantly different")
    
    return resilience_scores