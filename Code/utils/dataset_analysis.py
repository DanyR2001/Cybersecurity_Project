#!/usr/bin/env python3
"""
EMBER Dataset Comprehensive Analyzer
Genera analisi completa: statistiche, correlazioni, feature quality, pulizia
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Stile grafici
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10
})


class EMBERDatasetAnalyzer:
    """Analizzatore completo per dataset EMBER"""
    
    def __init__(self, data_dir, output_dir="Results/dataset_analysis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.report = {
            'basic_stats': {},
            'feature_quality': {},
            'correlation_analysis': {},
            'class_distribution': {},
            'feature_selection_impact': {}
        }
    
    def load_data(self):
        """Carica dataset EMBER"""
        print("\n" + "="*80)
        print("CARICAMENTO DATASET EMBER")
        print("="*80)
        
        import ember
        
        # Train
        print("\nCaricamento training set...")
        self.X_train, self.y_train = ember.read_vectorized_features(
            str(self.data_dir), subset="train"
        )
        train_mask = self.y_train != -1
        self.X_train = self.X_train[train_mask]
        self.y_train = self.y_train[train_mask]
        
        # Test
        print("Caricamento test set...")
        self.X_test, self.y_test = ember.read_vectorized_features(
            str(self.data_dir), subset="test"
        )
        test_mask = self.y_test != -1
        self.X_test = self.X_test[test_mask]
        self.y_test = self.y_test[test_mask]
        
        print(f"\n✓ Training: {self.X_train.shape[0]:,} samples × {self.X_train.shape[1]} features")
        print(f"✓ Test:     {self.X_test.shape[0]:,} samples × {self.X_test.shape[1]} features")
    
    def analyze_basic_stats(self):
        """Statistiche di base del dataset"""
        print("\n" + "="*80)
        print("STATISTICHE DI BASE")
        print("="*80)
        
        stats = {
            'train_samples': int(len(self.X_train)),
            'test_samples': int(len(self.X_test)),
            'total_features': int(self.X_train.shape[1]),
            'memory_size_mb': float((self.X_train.nbytes + self.X_test.nbytes) / (1024**2))
        }
        
        # Distribuzione classi
        train_benign = int(np.sum(self.y_train == 0))
        train_malware = int(np.sum(self.y_train == 1))
        test_benign = int(np.sum(self.y_test == 0))
        test_malware = int(np.sum(self.y_test == 1))
        
        stats['train_benign'] = train_benign
        stats['train_malware'] = train_malware
        stats['test_benign'] = test_benign
        stats['test_malware'] = test_malware
        stats['train_imbalance_ratio'] = float(train_malware / train_benign)
        stats['test_imbalance_ratio'] = float(test_malware / test_benign)
        
        self.report['basic_stats'] = stats
        
        print(f"\n Dataset Size:")
        print(f"   Training:   {stats['train_samples']:>10,} samples")
        print(f"   Test:       {stats['test_samples']:>10,} samples")
        print(f"   Features:   {stats['total_features']:>10,}")
        print(f"   Memory:     {stats['memory_size_mb']:>10,.1f} MB")
        
        print(f"\n Class Distribution:")
        print(f"   Training:")
        print(f"      Benign:  {train_benign:>10,} ({train_benign/len(self.y_train)*100:5.2f}%)")
        print(f"      Malware: {train_malware:>10,} ({train_malware/len(self.y_train)*100:5.2f}%)")
        print(f"      Ratio:   {stats['train_imbalance_ratio']:>10.3f} (malware/benign)")
        
        print(f"   Test:")
        print(f"      Benign:  {test_benign:>10,} ({test_benign/len(self.y_test)*100:5.2f}%)")
        print(f"      Malware: {test_malware:>10,} ({test_malware/len(self.y_test)*100:5.2f}%)")
        print(f"      Ratio:   {stats['test_imbalance_ratio']:>10.3f} (malware/benign)")
        
        return stats
    
    def analyze_feature_quality(self, sample_size=50000):
        """Analizza qualità delle feature"""
        print("\n" + "="*80)
        print(f"ANALISI QUALITÀ FEATURE (sample: {sample_size:,})")
        print("="*80)
        
        # Subsample per velocità
        sample_idx = np.random.choice(len(self.X_train), 
                                     min(sample_size, len(self.X_train)), 
                                     replace=False)
        X_sample = self.X_train[sample_idx]
        
        n_features = X_sample.shape[1]
        
        # 1. Feature costanti (zero variance)
        print("\n[1/6] Analisi feature costanti...")
        variances = np.var(X_sample, axis=0)
        constant_features = np.sum(variances == 0)
        near_constant = np.sum(variances < 1e-6)
        
        # 2. Feature sparse (mostly zeros)
        print("[2/6] Analisi sparsità...")
        zero_counts = np.sum(X_sample == 0, axis=0)
        sparsity = zero_counts / len(X_sample)
        very_sparse = np.sum(sparsity > 0.99)  # >99% zeri
        highly_sparse = np.sum(sparsity > 0.95)  # >95% zeri
        
        # 3. Missing values
        print("[3/6] Analisi valori mancanti...")
        nan_counts = np.sum(np.isnan(X_sample), axis=0)
        features_with_nan = np.sum(nan_counts > 0)
        
        # 4. Outliers (IQR method)
        print("[4/6] Analisi outlier...")
        outlier_features = 0
        for i in range(min(100, n_features)):  # Check first 100 features
            Q1 = np.percentile(X_sample[:, i], 25)
            Q3 = np.percentile(X_sample[:, i], 75)
            IQR = Q3 - Q1
            outliers = np.sum((X_sample[:, i] < Q1 - 3*IQR) | 
                            (X_sample[:, i] > Q3 + 3*IQR))
            if outliers > len(X_sample) * 0.05:  # >5% outliers
                outlier_features += 1
        
        # 5. Range e scale
        print("[5/6] Analisi range valori...")
        mins = np.min(X_sample, axis=0)
        maxs = np.max(X_sample, axis=0)
        ranges = maxs - mins
        
        different_scales = np.sum(ranges > 1000)  # Large scale differences
        
        # 6. Distribution types
        print("[6/6] Analisi distribuzioni...")
        skewed_features = 0
        for i in range(min(100, n_features)):
            if variances[i] > 0:
                skew = stats.skew(X_sample[:, i])
                if abs(skew) > 2:  # Highly skewed
                    skewed_features += 1
        
        quality = {
            'constant_features': int(constant_features),
            'near_constant_features': int(near_constant),
            'very_sparse_features': int(very_sparse),
            'highly_sparse_features': int(highly_sparse),
            'features_with_nan': int(features_with_nan),
            'different_scale_features': int(different_scales),
            'high_outlier_features': int(outlier_features),
            'highly_skewed_features': int(skewed_features),
            'avg_sparsity': float(np.mean(sparsity)),
            'avg_variance': float(np.mean(variances)),
            'min_value': float(np.min(mins)),
            'max_value': float(np.max(maxs))
        }
        
        self.report['feature_quality'] = quality
        
        print(f"\n Feature Quality Report:")
        print(f"\n   Problematic Features:")
        print(f"      Costanti (var=0):        {constant_features:>6,} ({constant_features/n_features*100:5.2f}%)")
        print(f"      Near-costanti (var<1e-6): {near_constant:>6,} ({near_constant/n_features*100:5.2f}%)")
        print(f"      Very sparse (>99% zero):  {very_sparse:>6,} ({very_sparse/n_features*100:5.2f}%)")
        print(f"      Highly sparse (>95% zero):{highly_sparse:>6,} ({highly_sparse/n_features*100:5.2f}%)")
        print(f"      Con NaN:                  {features_with_nan:>6,} ({features_with_nan/n_features*100:5.2f}%)")
        print(f"      Scala diversa (range>1k): {different_scales:>6,} ({different_scales/n_features*100:5.2f}%)")
        print(f"      Molti outlier (>5%):      {outlier_features:>6,} (su primi 100 testati)")
        print(f"      Highly skewed (|skew|>2): {skewed_features:>6,} (su primi 100 testati)")
        
        print(f"\n   Overall Statistics:")
        print(f"      Sparsità media:       {quality['avg_sparsity']*100:>6.2f}%")
        print(f"      Varianza media:       {quality['avg_variance']:>10.4f}")
        print(f"      Range valori:         [{quality['min_value']:>10.2f}, {quality['max_value']:>10.2f}]")
        
        # Suggerimenti
        print(f"\n    Raccomandazioni Cleaning:")
        recommendations = []
        if constant_features > 0:
            recommendations.append(f"Rimuovi {constant_features} feature costanti")
        if very_sparse > n_features * 0.1:
            recommendations.append(f"Considera rimozione {very_sparse} feature molto sparse")
        if features_with_nan > 0:
            recommendations.append(f"Gestisci {features_with_nan} feature con NaN (impute/remove)")
        if different_scales > n_features * 0.1:
            recommendations.append("Normalizza/scala feature (StandardScaler)")
        
        if recommendations:
            for rec in recommendations:
                print(f"      • {rec}")
        else:
            print(f"      ✓ Dataset già in buone condizioni!")
        
        return quality
    
    def analyze_correlations(self, sample_size=50000, threshold=0.95):
        """Analizza correlazioni tra feature"""
        print("\n" + "="*80)
        print(f"ANALISI CORRELAZIONI (sample: {sample_size:,})")
        print("="*80)
        
        # Subsample
        sample_idx = np.random.choice(len(self.X_train), 
                                     min(sample_size, len(self.X_train)), 
                                     replace=False)
        X_sample = self.X_train[sample_idx]
        
        # Rimuovi feature costanti per evitare warning
        variances = np.var(X_sample, axis=0)
        non_constant_mask = variances > 0
        X_sample_filtered = X_sample[:, non_constant_mask]
        
        print(f"\n[*] Calcolo matrice di correlazione...")
        print(f"    Features attive: {X_sample_filtered.shape[1]} (rimosse {np.sum(~non_constant_mask)} costanti)")
        
        # Calcola correlazione
        corr_matrix = np.corrcoef(X_sample_filtered.T)
        
        # Sostituisci NaN (per sicurezza)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Statistiche
        n_features = corr_matrix.shape[0]
        
        # Coppie altamente correlate
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        # Distribuzione correlazioni
        upper_tri = np.triu(corr_matrix, k=1)
        all_corrs = upper_tri[np.triu_indices_from(upper_tri, k=1)]
        
        corr_stats = {
            'total_features_analyzed': int(n_features),
            'high_corr_pairs_count': len(high_corr_pairs),
            'threshold': threshold,
            'avg_correlation': float(np.mean(np.abs(all_corrs))),
            'max_correlation': float(np.max(np.abs(all_corrs))),
            'corr_distribution': {
                'very_high_>0.95': int(np.sum(np.abs(all_corrs) > 0.95)),
                'high_0.8-0.95': int(np.sum((np.abs(all_corrs) > 0.8) & (np.abs(all_corrs) <= 0.95))),
                'moderate_0.5-0.8': int(np.sum((np.abs(all_corrs) > 0.5) & (np.abs(all_corrs) <= 0.8))),
                'low_<0.5': int(np.sum(np.abs(all_corrs) <= 0.5))
            }
        }
        
        self.report['correlation_analysis'] = corr_stats
        
        print(f"\n  Correlation Analysis:")
        print(f"   Features analizzate:       {n_features:>10,}")
        print(f"   Coppie altamente correlate: {len(high_corr_pairs):>10,} (|r| > {threshold})")
        print(f"   Correlazione media:        {corr_stats['avg_correlation']:>10.4f}")
        print(f"   Correlazione max:          {corr_stats['max_correlation']:>10.4f}")
        
        print(f"\n   Distribuzione Correlazioni:")
        total_pairs = len(all_corrs)
        for level, count in corr_stats['corr_distribution'].items():
            pct = count / total_pairs * 100 if total_pairs > 0 else 0
            print(f"      {level:20s}: {count:>10,} ({pct:5.2f}%)")
        
        # Plot heatmap (subsample per visualizzazione)
        print(f"\n[*] Generazione heatmap correlazioni...")
        
        # Prendi solo le prime 100 feature per visualizzazione
        n_vis = min(100, n_features)
        corr_vis = corr_matrix[:n_vis, :n_vis]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Heatmap completa (100x100)
        ax = axes[0]
        sns.heatmap(corr_vis, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title(f'Correlation Matrix (first {n_vis} features)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        # Distribuzione correlazioni
        ax = axes[1]
        ax.hist(all_corrs, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Threshold: {threshold}')
        ax.axvline(-threshold, color='red', linestyle='--', linewidth=2)
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Correlation Coefficient', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Pairwise Correlations', 
                    fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'correlation_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Salvato: {save_path}")
        plt.close()
        
        return corr_stats, corr_matrix
    
    def analyze_feature_importance(self, sample_size=50000, top_k=50):
        """Analizza importanza feature con Mutual Information"""
        print("\n" + "="*80)
        print(f"ANALISI IMPORTANZA FEATURE (sample: {sample_size:,})")
        print("="*80)
        
        # Subsample
        sample_idx = np.random.choice(len(self.X_train), 
                                     min(sample_size, len(self.X_train)), 
                                     replace=False)
        X_sample = self.X_train[sample_idx]
        y_sample = self.y_train[sample_idx]
        
        print(f"\n[*] Calcolo Mutual Information...")
        mi_scores = mutual_info_classif(
            X_sample, y_sample, 
            discrete_features=False,
            random_state=42,
            n_jobs=4
        )
        
        # Statistiche
        top_indices = np.argsort(mi_scores)[::-1][:top_k]
        
        importance_stats = {
            'mi_mean': float(np.mean(mi_scores)),
            'mi_std': float(np.std(mi_scores)),
            'mi_max': float(np.max(mi_scores)),
            'mi_min': float(np.min(mi_scores)),
            'zero_mi_features': int(np.sum(mi_scores == 0)),
            'top_features': top_indices.tolist()
        }
        
        self.report['feature_importance'] = importance_stats
        
        print(f"\n  Feature Importance (Mutual Information):")
        print(f"   MI medio:               {importance_stats['mi_mean']:>10.6f}")
        print(f"   MI std:                 {importance_stats['mi_std']:>10.6f}")
        print(f"   MI max:                 {importance_stats['mi_max']:>10.6f}")
        print(f"   MI min:                 {importance_stats['mi_min']:>10.6f}")
        print(f"   Feature con MI=0:       {importance_stats['zero_mi_features']:>10,}")
        
        # Plot
        print(f"\n[*] Generazione grafici importanza...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top-K feature importances
        ax = axes[0, 0]
        top_scores = mi_scores[top_indices]
        ax.barh(range(len(top_scores)), top_scores, color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(top_scores)))
        ax.set_yticklabels([f'F{i}' for i in top_indices], fontsize=8)
        ax.set_xlabel('Mutual Information Score', fontweight='bold')
        ax.set_title(f'Top-{top_k} Most Important Features', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # 2. MI distribution
        ax = axes[0, 1]
        ax.hist(mi_scores, bins=100, alpha=0.7, edgecolor='black', color='coral')
        ax.axvline(np.mean(mi_scores), color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(mi_scores):.4f}')
        ax.axvline(np.median(mi_scores), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(mi_scores):.4f}')
        ax.set_xlabel('Mutual Information Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of MI Scores', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative importance
        ax = axes[1, 0]
        sorted_mi = np.sort(mi_scores)[::-1]
        cumsum_mi = np.cumsum(sorted_mi) / np.sum(sorted_mi)
        ax.plot(range(len(cumsum_mi)), cumsum_mi * 100, linewidth=2, color='darkgreen')
        ax.axhline(80, color='red', linestyle='--', linewidth=1, label='80% threshold')
        ax.axhline(95, color='orange', linestyle='--', linewidth=1, label='95% threshold')
        
        # Trova numero feature per 80% e 95%
        n_80 = np.argmax(cumsum_mi >= 0.80) + 1
        n_95 = np.argmax(cumsum_mi >= 0.95) + 1
        ax.axvline(n_80, color='red', linestyle=':', alpha=0.5)
        ax.axvline(n_95, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Number of Features (sorted by importance)', fontweight='bold')
        ax.set_ylabel('Cumulative Importance (%)', fontweight='bold')
        ax.set_title('Cumulative Feature Importance', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.text(n_80, 82, f'{n_80} features', fontsize=9, ha='center', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(n_95, 97, f'{n_95} features', fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. MI score rank
        ax = axes[1, 1]
        ax.plot(range(len(mi_scores)), sorted_mi, linewidth=2, color='purple')
        ax.set_xlabel('Feature Rank', fontweight='bold')
        ax.set_ylabel('Mutual Information Score', fontweight='bold')
        ax.set_title('MI Score by Rank (Sorted)', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        save_path = self.output_dir / 'feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Salvato: {save_path}")
        plt.close()
        
        print(f"\n    Insight:")
        print(f"      • {n_80} features spiegano 80% dell'importanza")
        print(f"      • {n_95} features spiegano 95% dell'importanza")
        print(f"      • Consiglio: usa {n_80}-{n_95} feature per bilanciare performance/complessità")
        
        return importance_stats
    
    def simulate_feature_selection_impact(self, thresholds=[0.90, 0.95, 0.98, 0.99]):
        """Simula impatto di diversi threshold di correlazione"""
        print("\n" + "="*80)
        print("SIMULAZIONE IMPATTO FEATURE SELECTION")
        print("="*80)
        
        sample_size = 50000
        sample_idx = np.random.choice(len(self.X_train), 
                                     min(sample_size, len(self.X_train)), 
                                     replace=False)
        X_sample = self.X_train[sample_idx]
        
        # Rimuovi costanti
        variances = np.var(X_sample, axis=0)
        non_constant_mask = variances > 0
        X_filtered = X_sample[:, non_constant_mask]
        n_features_start = X_filtered.shape[1]
        
        print(f"\n[*] Features iniziali (dopo rimozione costanti): {n_features_start:,}")
        
        results = []
        
        for threshold in thresholds:
            print(f"\n[*] Testing threshold: {threshold}")
            
            # Calcola correlazioni
            corr_matrix = np.corrcoef(X_filtered.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Trova feature da rimuovere
            to_drop = set()
            for i in range(corr_matrix.shape[0]):
                for j in range(i+1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > threshold:
                        # Rimuovi la feature con varianza minore
                        var_i = variances[non_constant_mask][i]
                        var_j = variances[non_constant_mask][j]
                        if var_i < var_j:
                            to_drop.add(i)
                        else:
                            to_drop.add(j)
            
            n_remaining = n_features_start - len(to_drop)
            reduction_pct = len(to_drop) / n_features_start * 100
            
            results.append({
                'threshold': threshold,
                'features_removed': len(to_drop),
                'features_remaining': n_remaining,
                'reduction_pct': reduction_pct
            })
            
            print(f"   Rimosse: {len(to_drop):>6,} features ({reduction_pct:>5.2f}%)")
            print(f"   Rimaste: {n_remaining:>6,} features")
        
        self.report['feature_selection_impact'] = results
        
        # Plot
        print(f"\n[*] Generazione grafico impatto...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        thresholds_plot = [r['threshold'] for r in results]
        remaining_plot = [r['features_remaining'] for r in results]
        removed_plot = [r['features_removed'] for r in results]
        
        # Feature remaining vs threshold
        ax = axes[0]
        ax.plot(thresholds_plot, remaining_plot, marker='o', linewidth=3, 
               markersize=10, color='darkgreen', label='Remaining')
        ax.plot(thresholds_plot, removed_plot, marker='s', linewidth=3,
               markersize=10, color='darkred', label='Removed')
        ax.axhline(n_features_start, color='blue', linestyle='--', linewidth=2,
                  label=f'Original: {n_features_start}', alpha=0.5)
        ax.set_xlabel('Correlation Threshold', fontweight='bold', fontsize=13)
        ax.set_ylabel('Number of Features', fontweight='bold', fontsize=13)
        ax.set_title('Feature Selection Impact', fontweight='bold', fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        for r in results:
            ax.annotate(f"{r['features_remaining']:,}", 
                       xy=(r['threshold'], r['features_remaining']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
        
        # Reduction percentage
        ax = axes[1]
        reduction_plot = [r['reduction_pct'] for r in results]
        bars = ax.bar(range(len(thresholds_plot)), reduction_plot, 
                     color='coral', edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(thresholds_plot)))
        ax.set_xticklabels([f'{t:.2f}' for t in thresholds_plot], fontsize=12)
        ax.set_xlabel('Correlation Threshold', fontweight='bold', fontsize=13)
        ax.set_ylabel('Reduction (%)', fontweight='bold', fontsize=13)
        ax.set_title('Feature Reduction by Threshold', fontweight='bold', fontsize=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, r) in enumerate(zip(bars, results)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{r["reduction_pct"]:.1f}%\n({r["features_removed"]:,})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'feature_selection_impact.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Salvato: {save_path}")
        plt.close()
        
        return results
    
    def generate_summary_report(self):
        """Genera report JSON completo"""
        print("\n" + "="*80)
        print("GENERAZIONE REPORT FINALE")
        print("="*80)
        
        report_path = self.output_dir / 'dataset_analysis_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"\n✓ Report salvato: {report_path}")
        
        # Summary markdown
        md_path = self.output_dir / 'ANALYSIS_SUMMARY.md'
        
        with open(md_path, 'w') as f:
            f.write("# EMBER Dataset Analysis Report\n\n")
            
            f.write("##  Basic Statistics\n\n")
            stats = self.report['basic_stats']
            f.write(f"- **Training samples**: {stats['train_samples']:,}\n")
            f.write(f"- **Test samples**: {stats['test_samples']:,}\n")
            f.write(f"- **Total features**: {stats['total_features']}\n")
            f.write(f"- **Memory size**: {stats['memory_size_mb']:.1f} MB\n")
            f.write(f"- **Train imbalance ratio**: {stats['train_imbalance_ratio']:.3f}\n\n")
            
            f.write("##  Feature Quality\n\n")
            quality = self.report['feature_quality']
            f.write(f"- **Constant features**: {quality['constant_features']:,}\n")
            f.write(f"- **Very sparse (>99% zero)**: {quality['very_sparse_features']:,}\n")
            f.write(f"- **Features with NaN**: {quality['features_with_nan']:,}\n")
            f.write(f"- **Average sparsity**: {quality['avg_sparsity']*100:.2f}%\n\n")
            
            f.write("##  Correlation Analysis\n\n")
            corr = self.report['correlation_analysis']
            f.write(f"- **Features analyzed**: {corr['total_features_analyzed']:,}\n")
            f.write(f"- **High correlation pairs (>{corr['threshold']})**: {corr['high_corr_pairs_count']:,}\n")
            f.write(f"- **Average correlation**: {corr['avg_correlation']:.4f}\n\n")
            
            f.write("##  Feature Selection Impact\n\n")
            f.write("| Threshold | Removed | Remaining | Reduction % |\n")
            f.write("|-----------|---------|-----------|-------------|\n")
            for r in self.report['feature_selection_impact']:
                f.write(f"| {r['threshold']:.2f} | {r['features_removed']:,} | "
                       f"{r['features_remaining']:,} | {r['reduction_pct']:.1f}% |\n")
            
            f.write("\n##  Generated Files\n\n")
            f.write("- `correlation_analysis.png` - Correlation heatmap and distribution\n")
            f.write("- `feature_importance.png` - Mutual Information analysis\n")
            f.write("- `feature_selection_impact.png` - Impact of correlation thresholds\n")
            f.write("- `dataset_analysis_report.json` - Complete analysis data\n")
        
        print(f"✓ Summary salvato: {md_path}")
        
        # Print summary
        print("\n" + "="*80)
        print(" ANALISI COMPLETATA - SUMMARY")
        print("="*80)
        
        print(f"\n  Key Findings:")
        print(f"   • {stats['total_features']} features totali")
        print(f"   • {quality['constant_features']} feature costanti (da rimuovere)")
        print(f"   • {quality['very_sparse_features']} feature molto sparse")
        print(f"   • {corr['high_corr_pairs_count']:,} coppie altamente correlate")
        
        if 'feature_importance' in self.report:
            imp = self.report['feature_importance']
            print(f"   • {imp['zero_mi_features']} feature con MI=0 (inutili)")
        
        print(f"\n Raccomandazioni:")
        print(f"   • Rimuovi {quality['constant_features']} feature costanti")
        
        # Trova threshold ottimale
        best_threshold = None
        for r in self.report['feature_selection_impact']:
            if r['reduction_pct'] > 10 and r['reduction_pct'] < 40:  # Sweet spot
                best_threshold = r
                break
        
        if best_threshold:
            print(f"   • Usa threshold {best_threshold['threshold']:.2f} per correlazione")
            print(f"     → Rimuove {best_threshold['features_removed']:,} feature ({best_threshold['reduction_pct']:.1f}%)")
            print(f"     → Rimangono {best_threshold['features_remaining']:,} feature")
        
        print(f"\n Output directory: {self.output_dir}/")
        print(f"   • dataset_analysis_report.json")
        print(f"   • ANALYSIS_SUMMARY.md")
        print(f"   • correlation_analysis.png")
        print(f"   • feature_importance.png")
        print(f"   • feature_selection_impact.png")
        
        print("\n" + "="*80)
    
    def run_complete_analysis(self):
        sample_size=250000
        """Esegue analisi completa"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " "*20 + "EMBER DATASET ANALYZER" + " "*36 + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        # 1. Carica dati
        self.load_data()
        
        # 2. Statistiche base
        self.analyze_basic_stats()
        
        # 3. Qualità feature
        self.analyze_feature_quality(sample_size=sample_size)
        
        # 4. Correlazioni
        self.analyze_correlations(sample_size=sample_size, threshold=0.95)
        
        # 5. Importanza feature
        self.analyze_feature_importance(sample_size=sample_size, top_k=50)
        
        # 6. Impatto feature selection
        self.simulate_feature_selection_impact(thresholds=[0.90, 0.95, 0.98, 0.99])
        
        # 7. Report finale
        self.generate_summary_report()


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "Dataset/ember2018"
        print(f"Usando dataset default: {data_dir}")
        print(f"Usage: python analyze_dataset.py <path_to_ember_dataset>\n")
    
    # Crea analyzer e esegui
    analyzer = EMBERDatasetAnalyzer(data_dir)
    analyzer.run_complete_analysis()
    
    print("\n Analisi completata con successo!")


if __name__ == "__main__":
    main()