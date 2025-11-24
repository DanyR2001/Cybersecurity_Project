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
        OTTIMIZZATO: usa sampling per calcolare threshold su grandi modelli
        
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
        
        # FIX: Per modelli grandi, usa SAMPLING per calcolare threshold
        all_weights_cat = torch.cat(all_weights)
        n_weights = all_weights_cat.numel()
        
        # Se tensore troppo grande (>10M elementi), usa sampling
        MAX_SAMPLE_SIZE = 10_000_000  # 10M elements max
        
        if n_weights > MAX_SAMPLE_SIZE:
            # Sample random subset per calcolare threshold
            sample_indices = torch.randperm(n_weights)[:MAX_SAMPLE_SIZE]
            weights_sample = all_weights_cat[sample_indices]
            threshold = torch.quantile(weights_sample, pruning_rate)
            print(f"  [Memory optimization] Sampled {MAX_SAMPLE_SIZE:,} / {n_weights:,} weights for threshold")
        else:
            # FIX alternativo: usa numpy per grandi tensori
            try:
                threshold = torch.quantile(all_weights_cat, pruning_rate)
            except RuntimeError:
                # Fallback: converti a numpy
                weights_np = all_weights_cat.cpu().numpy()
                threshold_np = np.quantile(weights_np, pruning_rate)
                threshold = torch.tensor(threshold_np, device=all_weights_cat.device)
                print(f"  [Fallback] Used numpy for quantile calculation")
        
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
        
        # DEBUG: Stampa distribuzione
        print(f"\n🔍 Stability Score Distribution:")
        print(f"  Min:  {np.min(stability_scores):.4f}")
        print(f"  10%:  {np.percentile(stability_scores, 10):.4f}")
        print(f"  50%:  {np.median(stability_scores):.4f}")
        print(f"  90%:  {np.percentile(stability_scores, 90):.4f}")
        print(f"  Max:  {np.max(stability_scores):.4f}")

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




# Esempio di utilizzo
if __name__ == "__main__":
    print("Weight Pruning Detector per Poisoning Detection")
    print("Usa questo modulo in main.py per la detection")