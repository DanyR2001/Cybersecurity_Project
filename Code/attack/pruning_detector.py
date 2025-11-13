#!/usr/bin/env python3
"""
Poisoning Detection tramite Weight Pruning Progressivo
Idea: Il poisoning crea segnali forti su pochi pesi → i pesi piccoli sono noise
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy


class WeightPruningDetector:
    """
    Detector basato su pruning progressivo dei pesi più piccoli.
    
    Logica:
    1. Ordina tutti i pesi del modello per valore assoluto
    2. Pruning progressivo: azzera % crescente dei pesi più piccoli
    3. Per ogni livello di pruning, testa quali campioni mantengono la predizione
    4. Campioni avvelenati = perdono predizione prima (dipendono da pesi specifici)
    5. Campioni puliti = mantengono predizione più a lungo (pattern generali)
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.original_state = copy.deepcopy(model.state_dict())
    
    def prune_smallest_weights(self, pruning_rate, target_layers=None):
        """
        Crea COPIA del modello con pruning_rate% dei pesi più piccoli azzerati
        
        Args:
            pruning_rate: percentuale di pesi da azzerare (0.0 - 1.0)
            target_layers: lista layer da prunare (None = tutti)
        
        Returns:
            pruned_model, pruning_stats
        """
        pruned_model = copy.deepcopy(self.model)
        
        # Colleziona tutti i pesi da considerare
        all_weights = []
        weight_params = []
        
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                should_prune = False
                
                if target_layers is not None:
                    should_prune = any(layer in name for layer in target_layers)
                else:
                    should_prune = 'weight' in name  # Solo pesi, non bias
                
                if should_prune and param.requires_grad:
                    all_weights.append(param.abs().flatten())
                    weight_params.append((name, param))
        
        # Concatena tutti i pesi e trova threshold
        all_weights_cat = torch.cat(all_weights)
        threshold = torch.quantile(all_weights_cat, pruning_rate)
        
        # Applica pruning
        n_total = 0
        n_pruned = 0
        
        with torch.no_grad():
            for name, param in weight_params:
                mask = param.abs() > threshold
                n_total += param.numel()
                n_pruned += (~mask).sum().item()
                param.mul_(mask.float())
        
        pruning_stats = {
            'pruning_rate_requested': pruning_rate,
            'pruning_rate_actual': n_pruned / n_total if n_total > 0 else 0,
            'threshold': float(threshold.cpu()),
            'n_pruned': n_pruned,
            'n_total': n_total,
            'n_remaining': n_total - n_pruned
        }
        
        return pruned_model, pruning_stats
    
    def compute_stability_scores(self, X, batch_size=256, 
                                 pruning_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                 verbose=True):
        """
        Calcola stability score per ogni campione attraverso pruning progressivo
        
        Stability score = frazione di pruning levels dove predizione rimane uguale all'originale
        
        Returns:
            stability_scores: array [N] con score per ogni campione
            prediction_matrix: array [N, n_pruning_levels] con predizioni
            pruning_stats_list: lista di pruning stats
        """
        print(f"\n=== Weight Pruning Stability Analysis ===")
        print(f"Campioni: {len(X)}")
        print(f"Pruning levels: {len(pruning_rates)}")
        
        # Dataset
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 1. Predizione originale (no pruning)
        self.model.eval()
        original_preds = []
        
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                original_preds.extend(preds)
        
        original_preds = np.array(original_preds)
        
        # 2. Predizioni con pruning progressivo
        prediction_matrix = np.zeros((len(X), len(pruning_rates)), dtype=int)
        pruning_stats_list = []
        
        iterator = enumerate(pruning_rates)
        if verbose:
            iterator = tqdm(list(iterator), desc="Pruning Levels")
        
        for prune_idx, pruning_rate in iterator:
            # Crea modello con pruning
            pruned_model, stats = self.prune_smallest_weights(pruning_rate)
            pruned_model.eval()
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'rate': f"{pruning_rate:.1%}",
                    'pruned': f"{stats['n_pruned']:,}"
                })
            
            pruning_stats_list.append(stats)
            
            # Predici con modello pruned
            pruned_preds = []
            with torch.no_grad():
                for (batch_x,) in loader:
                    batch_x = batch_x.to(self.device)
                    logits = pruned_model(batch_x)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    pruned_preds.extend(preds)
            
            prediction_matrix[:, prune_idx] = np.array(pruned_preds)
            
            del pruned_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 3. Calcola stability scores
        # Stability = quante volte la predizione rimane uguale all'originale
        stability_scores = np.mean(
            prediction_matrix == original_preds[:, np.newaxis], 
            axis=1
        )
        
        print(f"\n Stability Statistics:")
        print(f"  Mean: {np.mean(stability_scores):.4f}")
        print(f"  Std:  {np.std(stability_scores):.4f}")
        print(f"  Min:  {np.min(stability_scores):.4f}")
        print(f"  Max:  {np.max(stability_scores):.4f}")
        
        # Analisi distribuzione
        low_stab = np.sum(stability_scores < 0.3)
        mid_stab = np.sum((stability_scores >= 0.3) & (stability_scores <= 0.7))
        high_stab = np.sum(stability_scores > 0.7)
        
        print(f"\n Distribuzione Stability:")
        print(f"  Bassa (< 0.3):   {low_stab:6d} ({low_stab/len(X)*100:.1f}%)")
        print(f"  Media (0.3-0.7):  {mid_stab:6d} ({mid_stab/len(X)*100:.1f}%)")
        print(f"  Alta  (> 0.7):    {high_stab:6d} ({high_stab/len(X)*100:.1f}%)")
        
        return stability_scores, original_preds, prediction_matrix, pruning_stats_list
    
    def detect_poisoned_samples(self, X, y, poison_indices=None,
                               pruning_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                               threshold=None,
                               batch_size=256):
        """
        Identifica campioni avvelenati tramite stability analysis
        
        Ipotesi: campioni avvelenati hanno BASSA stability (predizione cambia subito con pruning)
        """
        # Calcola stability scores
        stability_scores, predictions, pred_matrix, pruning_stats = self.compute_stability_scores(
            X, batch_size=batch_size, pruning_rates=pruning_rates
        )
        
        # Auto-calcola threshold (inverso rispetto a resilience!)
        # BASSA stability = sospetto avvelenato
        if threshold is None:
            # Usa 10° percentile (i più instabili)
            threshold = np.percentile(stability_scores, 10)
            print(f"\n Threshold (10th percentile - low stability): {threshold:.4f}")
        
        # Identifica suspected poisoned: stability SOTTO threshold
        suspected_poison = stability_scores < threshold
        
        results = {
            'stability_scores': stability_scores,
            'predictions': predictions,
            'prediction_matrix': pred_matrix,
            'pruning_stats': pruning_stats,
            'suspected_poison_indices': np.where(suspected_poison)[0],
            'threshold': threshold,
            'n_suspected': np.sum(suspected_poison),
            'pruning_rates': pruning_rates
        }
        
        # Metriche con ground truth
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
                # INVERTI: bassa stability = high poisoning probability
                auc = roc_auc_score(poison_mask, -stability_scores)  # Nota il segno meno!
                ap = average_precision_score(poison_mask, -stability_scores)
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
            
            print(f"\n===  Detection Performance ===")
            print(f"Actual poisoned:  {len(poison_indices)}")
            print(f"Detected:         {np.sum(suspected_poison)}")
            print(f"\n Metriche:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC-ROC:   {auc:.4f}")
            
            # Stability per classe
            print(f"\n Stability by Class:")
            for label in [0, 1]:
                mask = y == label
                if np.sum(mask) > 0:
                    label_name = "Benign" if label == 0 else "Malware"
                    mean_stab = np.mean(stability_scores[mask])
                    print(f"  {label_name:8s}: {mean_stab:.4f} ± {np.std(stability_scores[mask]):.4f}")
            
            # Confronto poisoned vs clean
            poison_stab = stability_scores[poison_mask]
            clean_stab = stability_scores[~poison_mask]
            print(f"\n Stability Comparison:")
            print(f"  Poisoned: {np.mean(poison_stab):.4f} ± {np.std(poison_stab):.4f}")
            print(f"  Clean:    {np.mean(clean_stab):.4f} ± {np.std(clean_stab):.4f}")
            print(f"  Δ:        {np.mean(poison_stab) - np.mean(clean_stab):+.4f}")
            
            if np.mean(poison_stab) < np.mean(clean_stab):
                print(f"   Poisoned samples have LOWER stability (expected!)")
            else:
                print(f"    WARNING: Poisoned have HIGHER stability (unexpected)")
        
        return results
    
    def restore_original_weights(self):
        """Ripristina i pesi originali del modello"""
        self.model.load_state_dict(self.original_state)


def plot_pruning_detection_results(detection_results, save_path='pruning_detection_results.png'):
    """
    Visualizza risultati della detection con weight pruning
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    stability_scores = detection_results['stability_scores']
    threshold = detection_results['threshold']
    pruning_rates = detection_results['pruning_rates']
    
    # 1. Distribuzione stability scores
    ax = axes[0, 0]
    ax.hist(stability_scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold: {threshold:.3f}')
    ax.axvline(np.median(stability_scores), color='green', linestyle=':', 
               linewidth=2, label=f'Median: {np.median(stability_scores):.3f}')
    ax.set_xlabel('Stability Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Stability Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter stability vs index
    ax = axes[0, 1]
    indices = np.arange(len(stability_scores))
    
    if 'ground_truth' in detection_results:
        poison_mask = np.zeros(len(stability_scores), dtype=bool)
        poison_mask[detection_results['ground_truth']['poison_indices']] = True
        
        ax.scatter(indices[~poison_mask], stability_scores[~poison_mask], 
                  c='blue', alpha=0.3, s=1, label='Clean')
        ax.scatter(indices[poison_mask], stability_scores[poison_mask], 
                  c='red', alpha=0.6, s=3, label='Poisoned')
    else:
        ax.scatter(indices, stability_scores, c='blue', alpha=0.3, s=1)
    
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Stability Score')
    ax.set_title('Stability Scores by Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Pruning impact curve
    ax = axes[0, 2]
    pred_matrix = detection_results['prediction_matrix']
    original_preds = detection_results['predictions']
    
    # % campioni che mantengono predizione per ogni pruning level
    pct_maintained = []
    for i in range(len(pruning_rates)):
        pct = np.mean(pred_matrix[:, i] == original_preds) * 100
        pct_maintained.append(pct)
    
    ax.plot(np.array(pruning_rates) * 100, pct_maintained, marker='o', linewidth=2)
    ax.set_xlabel('Pruning Rate (%)')
    ax.set_ylabel('% Predictions Maintained')
    ax.set_title('Model Robustness to Pruning')
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5)
    
    # 4. Confusion matrix
    if 'ground_truth' in detection_results:
        ax = axes[1, 0]
        metrics = detection_results['ground_truth']['detection_metrics']
        
        cm = np.array([[metrics['true_negatives'], metrics['false_positives']],
                       [metrics['false_negatives'], metrics['true_positives']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Clean', 'Poisoned'],
                   yticklabels=['Clean', 'Poisoned'],
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Detection Confusion Matrix')
        
        # 5. Metriche
        ax = axes[1, 1]
        metric_names = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        metric_values = [
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auc_roc']
        ]
        
        bars = ax.bar(metric_names, metric_values, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Detection Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # 6. Stability distribution per class
        ax = axes[1, 2]
        poison_mask = np.zeros(len(stability_scores), dtype=bool)
        poison_mask[detection_results['ground_truth']['poison_indices']] = True
        
        clean_scores = stability_scores[~poison_mask]
        poison_scores = stability_scores[poison_mask]
        
        ax.hist(clean_scores, bins=30, alpha=0.6, label='Clean', density=True)
        ax.hist(poison_scores, bins=30, alpha=0.6, label='Poisoned', density=True, color='red')
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Stability Score')
        ax.set_ylabel('Density')
        ax.set_title('Stability Distribution: Clean vs Poisoned')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        for i in range(3):
            axes[1, i].axis('off')
        
        text = f"Detection Statistics:\n\n"
        text += f"Total samples: {len(stability_scores)}\n"
        text += f"Suspected: {detection_results['n_suspected']}\n"
        text += f"Threshold: {threshold:.4f}\n\n"
        text += f"Stats:\n"
        text += f"  Mean: {np.mean(stability_scores):.4f}\n"
        text += f"  Median: {np.median(stability_scores):.4f}\n"
        text += f"  Std: {np.std(stability_scores):.4f}"
        
        axes[1, 1].text(0.5, 0.5, text, ha='center', va='center', 
                       fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Grafico salvato: {save_path}")
    plt.close()


# Esempio di utilizzo
if __name__ == "__main__":
    print("Weight Pruning Detector per Poisoning Detection")
    print("Usa questo modulo in main.py per la detection")