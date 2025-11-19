#!/usr/bin/env python3
"""
Isolation Forest Defense - Baseline dal paper Severi et al.
Defense contro backdoor poisoning tramite anomaly detection su reduced feature space
"""

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


class IsolationForestDefender:
    """
    Implementa la defense Isolation Forest dal paper:
    "Explanation-Guided Backdoor Poisoning Attacks Against Malware Classifiers"
    
    Key insight: Isolation Forest funziona SOLO su reduced feature space
    """
    
    def __init__(self, contamination=0.01, n_top_features=32, random_state=42):
        """
        Args:
            contamination: Expected proportion of outliers (poison rate estimate)
            n_top_features: Number of most important features to use (paper usa 32)
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_top_features = n_top_features
        self.random_state = random_state
        self.top_features = None
        self.iso_forest = None
    
    def select_top_features_mi(self, X_train, y_train, verbose=True, max_samples=50000):
        """
        Seleziona top-k features usando Mutual Information
        (alternativa a SHAP feature importance)
        
        OTTIMIZZATO: Usa subset per ridurre memoria
        
        Args:
            max_samples: massimo numero di campioni da usare (default: 50000)
        
        Returns:
            top_feature_indices: array of top feature indices
        """
        print(f"\n=== Feature Selection (Mutual Information) ===")
        print(f"Computing MI for {X_train.shape[1]} features...")
        
        # RIDUZIONE MEMORIA: Usa subset
        n_samples = min(max_samples, len(X_train))
        if n_samples < len(X_train):
            print(f"  [Memory optimization] Using {n_samples:,} / {len(X_train):,} samples")
            sample_idx = np.random.choice(len(X_train), n_samples, replace=False)
            X_sample = X_train[sample_idx]
            y_sample = y_train[sample_idx]
        else:
            X_sample = X_train
            y_sample = y_train
        
        # Mutual Information
        mi_scores = mutual_info_classif(
            X_sample, y_sample, 
            discrete_features=False,
            random_state=self.random_state,
            n_jobs=4  # LIMITATO a 4 core invece di -1
        )
        
        # Top-k features
        top_indices = np.argsort(mi_scores)[::-1][:self.n_top_features]
        self.top_features = top_indices
        
        if verbose:
            print(f"Selected top-{self.n_top_features} features")
            print(f"  MI range: [{mi_scores[top_indices[-1]]:.4f}, {mi_scores[top_indices[0]]:.4f}]")
        
        return top_indices
    
    def select_top_features_shap(self, model, X_train, y_train, device, batch_size=256):
        """
        Seleziona top-k features usando SHAP values (come nel paper)
        Più accurato ma più lento
        """
        print(f"\n=== Feature Selection (SHAP) ===")
        print(f"Computing SHAP values... (this may take a while)")
        
        import shap
        
        # Usa subset per accelerare
        sample_size = min(1000, len(X_train))
        X_sample = X_train[np.random.choice(len(X_train), sample_size, replace=False)]
        
        # SHAP explainer
        def model_predict(x):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(device)
                logits = model(x_tensor)
                return torch.sigmoid(logits).cpu().numpy()
        
        explainer = shap.KernelExplainer(model_predict, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)
        
        # Feature importance = mean absolute SHAP
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Top-k features
        top_indices = np.argsort(feature_importance)[::-1][:self.n_top_features]
        self.top_features = top_indices
        
        print(f"Selected top-{self.n_top_features} features using SHAP")
        
        return top_indices
    
    def fit_detector(self, X_train, y_train, use_shap=False, model=None, device=None):
        """
        Fit Isolation Forest detector
        
        Args:
            X_train: training features
            y_train: training labels
            use_shap: if True, use SHAP for feature selection (slower but better)
            model: PyTorch model (required if use_shap=True)
            device: device for model (required if use_shap=True)
        """
        # Feature selection
        if use_shap:
            if model is None or device is None:
                raise ValueError("model and device required for SHAP-based selection")
            self.select_top_features_shap(model, X_train, y_train, device)
        else:
            self.select_top_features_mi(X_train, y_train)
        
        # Reduced feature space (SOLO benign samples)
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        X_benign_reduced = X_benign[:, self.top_features]
        
        print(f"\n=== Training Isolation Forest ===")
        print(f"Benign samples: {len(X_benign)}")
        print(f"Feature space: {X_benign_reduced.shape[1]} features")
        print(f"Contamination: {self.contamination}")
        
        # Fit Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iso_forest.fit(X_benign_reduced)
        
        print("Isolation Forest trained successfully")
    
    def detect_outliers(self, X_train, y_train):
        """
        Detect outliers in training set
        
        Returns:
            suspected_poison_mask: boolean mask (True = suspected poison)
            outlier_scores: anomaly scores for each sample
        """
        if self.iso_forest is None or self.top_features is None:
            raise ValueError("Must call fit_detector() first")
        
        # Reduced feature space (SOLO benign samples)
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        X_benign_reduced = X_benign[:, self.top_features]
        
        # Predict outliers (-1 = outlier, 1 = inlier)
        predictions = self.iso_forest.predict(X_benign_reduced)
        outlier_scores = self.iso_forest.decision_function(X_benign_reduced)
        
        # Create full mask
        suspected_poison_mask_benign = predictions == -1
        suspected_poison_mask = np.zeros(len(X_train), dtype=bool)
        suspected_poison_mask[benign_mask] = suspected_poison_mask_benign
        
        # Statistics
        n_outliers = np.sum(suspected_poison_mask)
        print(f"\n=== Outlier Detection Results ===")
        print(f"Total samples: {len(X_train)}")
        print(f"Benign samples analyzed: {len(X_benign)}")
        print(f"Detected outliers: {n_outliers} ({n_outliers/len(X_benign)*100:.2f}% of benign)")
        
        return suspected_poison_mask, outlier_scores
    
    def clean_dataset(self, X_train, y_train, poison_indices=None):
        """
        Remove suspected poisoned samples from training set
        
        Args:
            X_train, y_train: training data
            poison_indices: ground truth poison indices (for evaluation)
        
        Returns:
            X_clean, y_clean: cleaned dataset
            defense_metrics: dict with detection metrics
        """
        # Detect outliers
        suspected_poison_mask, outlier_scores = self.detect_outliers(X_train, y_train)
        
        # Clean dataset
        clean_mask = ~suspected_poison_mask
        X_clean = X_train[clean_mask]
        y_clean = y_train[clean_mask]
        
        print(f"\n=== Dataset Cleaning ===")
        print(f"Original size: {len(X_train)}")
        print(f"Cleaned size:  {len(X_clean)}")
        print(f"Removed:       {np.sum(suspected_poison_mask)}")
        
        # Evaluation metrics (if ground truth available)
        defense_metrics = {
            'n_removed': int(np.sum(suspected_poison_mask)),
            'n_remaining': int(np.sum(clean_mask)),
            'removal_rate': float(np.sum(suspected_poison_mask) / len(X_train))
        }
        
        if poison_indices is not None:
            poison_mask = np.zeros(len(X_train), dtype=bool)
            poison_mask[poison_indices] = True
            
            # Detection metrics
            tp = np.sum(poison_mask & suspected_poison_mask)
            fp = np.sum(~poison_mask & suspected_poison_mask)
            tn = np.sum(~poison_mask & ~suspected_poison_mask)
            fn = np.sum(poison_mask & ~suspected_poison_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            defense_metrics.update({
                'ground_truth': {
                    'n_poison': len(poison_indices),
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
            })
            
            print(f"\n=== Detection Metrics ===")
            print(f"Ground truth poisons: {len(poison_indices)}")
            print(f"Detected (TP): {tp} ({tp/len(poison_indices)*100:.1f}%)")
            print(f"Missed (FN):   {fn} ({fn/len(poison_indices)*100:.1f}%)")
            print(f"False alarms (FP): {fp}")
            print(f"\nPrecision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
        
        return X_clean, y_clean, defense_metrics


def plot_isolation_forest_results(outlier_scores, poison_indices, benign_indices, 
                                   save_path='isolation_forest_results.png'):
    """
    Visualizza risultati Isolation Forest
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Distribution of outlier scores
    ax = axes[0]
    
    if poison_indices is not None:
        poison_mask = np.isin(benign_indices, poison_indices)
        clean_mask = ~poison_mask
        
        ax.hist(outlier_scores[clean_mask], bins=50, alpha=0.6, 
                label='Clean', color='blue', density=True)
        ax.hist(outlier_scores[poison_mask], bins=50, alpha=0.6, 
                label='Poisoned', color='red', density=True)
        ax.legend()
    else:
        ax.hist(outlier_scores, bins=50, alpha=0.7, color='blue', density=True)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=2, 
               label='Decision Boundary (0)')
    ax.set_xlabel('Outlier Score (lower = more anomalous)')
    ax.set_ylabel('Density')
    ax.set_title('Isolation Forest Outlier Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax = axes[1]
    indices = np.arange(len(outlier_scores))
    
    if poison_indices is not None:
        poison_mask = np.isin(benign_indices, poison_indices)
        
        ax.scatter(indices[~poison_mask], outlier_scores[~poison_mask],
                  c='blue', alpha=0.3, s=1, label='Clean')
        ax.scatter(indices[poison_mask], outlier_scores[poison_mask],
                  c='red', alpha=0.6, s=3, label='Poisoned')
        ax.legend()
    else:
        ax.scatter(indices, outlier_scores, c='blue', alpha=0.3, s=1)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Outlier Score')
    ax.set_title('Outlier Scores by Sample')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nIsolation Forest plot salvato: {save_path}")
    plt.close()


# Test rapido
if __name__ == "__main__":
    print("Isolation Forest Defender per Backdoor Poisoning Detection")
    print("Usa questo modulo in main.py per la defense")