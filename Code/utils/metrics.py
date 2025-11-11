#!/usr/bin/env python3
"""
Modulo per calcolo metriche e valutazione modelli
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from tqdm import tqdm


class MetricsCalculator:
    """Calcola e salva tutte le metriche di valutazione"""

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Calcola tutte le metriche principali
        
        Args:
            y_true: etichette vere
            y_pred: predizioni binarie
            y_pred_proba: probabilità predette (opzionale, per AUC)
            
        Returns:
            dict con tutte le metriche
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred_proba))
            except ValueError:
                metrics['auc_roc'] = None

        # Confusion matrix (assume labels 0/1)
        cm = confusion_matrix(y_true, y_pred)
        
        # Gestisci caso con una sola classe predetta
        if cm.shape == (1, 1):
            if y_pred[0] == 0:
                # Tutto predetto come 0
                tn = cm[0, 0]
                fp = 0
                fn = int(np.sum(y_true == 1))
                tp = 0
            else:
                # Tutto predetto come 1
                tn = 0
                fp = int(np.sum(y_true == 0))
                fn = 0
                tp = cm[0, 0]
        else:
            tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)

        # Specificity (True Negative Rate)
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        return metrics

    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba, metric='f1', n_thresholds=100):
        """
        Trova la soglia ottimale per massimizzare una metrica
        
        Args:
            y_true: etichette vere
            y_pred_proba: probabilità predette
            metric: metrica da ottimizzare ('f1', 'accuracy', 'balanced_accuracy')
            n_thresholds: numero di threshold da testare
            
        Returns:
            optimal_threshold, best_metric_value
        """
        thresholds = np.linspace(0.1, 0.9, n_thresholds)
        best_metric = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            if metric == 'f1':
                current_metric = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                current_metric = accuracy_score(y_true, y_pred)
            elif metric == 'balanced_accuracy':
                # Media di recall per ogni classe
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                current_metric = (sensitivity + specificity) / 2
            else:
                raise ValueError(f"Metrica non supportata: {metric}")
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_threshold = threshold
        
        return float(best_threshold), float(best_metric)

    @staticmethod
    def evaluate_model(model, data_loader, device):
        """
        Valuta il modello e restituisce predizioni e metriche
        
        Args:
            model: modello PyTorch
            data_loader: DataLoader con dati di test
            device: device PyTorch (cpu/cuda/mps)
            
        Returns:
            metrics, all_targets, all_preds, all_probs
        """
        model.eval()
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating", leave=False):
                data = data.to(device).float()
                target = target.to(device).float()
                output = model(data)  # logits expected
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(float)

                all_targets.extend(target.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = MetricsCalculator.calculate_metrics(
            all_targets, all_preds, all_probs
        )

        return metrics, all_targets, all_preds, all_probs

    @staticmethod
    def print_metrics(metrics, title="Metrics"):
        """Stampa metriche in formato leggibile"""
        print(f"\n=== {title} ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")