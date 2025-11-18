#!/usr/bin/env python3
"""
Modulo per visualizzazione risultati e metriche
Con comparison plot migliorato: 4 grafici sopra (precision + 3 altri) + 4 confusion matrices sotto
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 


def plot_comparison(results, save_path="comparison_plot.png"):
    """
    Visualizza confronto tra modelli con metriche e confusion matrix
    FIX: Supporta sia 'poisoned' che 'backdoored' come chiave
    """
    scenario_configs = {
        'clean':    {'name': 'Clean',          'color': 'green',  'cmap': 'Greens'},
        'backdoored': {'name': 'Backdoored',   'color': 'red',    'cmap': 'Reds'},  
        'poisoned': {'name': 'Poisoned',       'color': 'red',    'cmap': 'Reds'},  
        'isolation_forest': {'name': 'IsoForest Defense', 'color': 'orange', 'cmap': 'Oranges'},
        'pruned':   {'name': 'Pruned Defense', 'color': 'blue',   'cmap': 'Blues'},
        'noisy':    {'name': 'Noisy Defense',  'color': 'purple', 'cmap': 'Purples'}
    }
    
    available = [k for k in scenario_configs if k in results and 'test' in results[k]]
    if len(available) < 2:
        print("[!] Non abbastanza scenari per il confronto")
        return
    
    scenario_data = {k: scenario_configs[k] for k in available}
    
    # Figura
    fig = plt.figure(figsize=(26, 15))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.4)
    
    names = [scenario_data[s]['name'] for s in available]
    fig.suptitle(f"Model Comparison: {' vs '.join(names)}", 
                 fontsize=24, fontweight='bold', y=0.96)
    
    # === RIGA 1: METRICHE A BARRE ===
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'accuracy']
    titles = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = fig.add_subplot(gs[0, idx])
        values = [results[s]['test'].get(metric, 0) for s in available]
        colors = [scenario_data[s]['color'] for s in available]
        
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, values, width=0.5, color=colors, alpha=0.85, 
                      edgecolor='black', linewidth=1.6)
        
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # === RIGA 2: CONFUSION MATRIX ===
    malware_labels = ['Benign', 'Malware']
    
    # FIX: Determina dinamicamente quali modelli mostrare
    # Priorità: clean, backdoored/poisoned, isolation_forest, pruned, noisy
    models_to_show = []
    if 'clean' in results:
        models_to_show.append('clean')
    
    # Usa 'backdoored' se disponibile, altrimenti 'poisoned'
    if 'backdoored' in results:
        models_to_show.append('backdoored')
    elif 'poisoned' in results:
        models_to_show.append('poisoned')
    
    # Aggiungi defenses disponibili
    for defense in ['isolation_forest', 'pruned', 'noisy']:
        if defense in results and len(models_to_show) < 4:
            models_to_show.append(defense)
    
    # Padding se necessario
    while len(models_to_show) < 4:
        models_to_show.append(None)
    
    for idx, key in enumerate(models_to_show):
        ax = fig.add_subplot(gs[1, idx])
        
        if key is not None and key in results and 'test' in results[key]:
            m = results[key]['test']
            cm = np.array([[m['true_negative'], m['false_positive']],
                           [m['false_negative'], m['true_positive']]])
            
            name = scenario_data[key]['name']
            cmap = scenario_data[key]['cmap']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                        square=True, linewidths=2.5, linecolor='black',
                        annot_kws={"size": 20, "weight": "bold"},
                        xticklabels=malware_labels,
                        yticklabels=malware_labels,
                        ax=ax)
            
            ax.set_title(f"{name}\nConfusion Matrix", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
            ax.set_ylabel('Actual Label', fontsize=13, fontweight='bold')
        else:
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"[+] Comparison plot aggiornato e salvato: {save_path}")
    plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"):
    """
    Visualizza l'andamento del training
    
    Args:
        train_losses: lista loss di training per epoca
        val_losses: lista loss di validazione per epoca
        train_accs: lista accuracy di training per epoca
        val_accs: lista accuracy di validazione per epoca
        save_path: percorso dove salvare il grafico
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Training history salvato in: {save_path}")
    plt.close()


def plot_correlation_matrix(X, save_path="correlation_matrix.png", method='pearson', threshold=0.8, 
                            figsize=(12, 10), sample_size=None, annot=False):
    """
    Calcola e salva la matrice di correlazione delle feature come heatmap.
    
    Args:
        X: array o DataFrame delle feature
        save_path: percorso dove salvare il grafico
        method: metodo di correlazione ('pearson', 'spearman', 'kendall')
        threshold: soglia assoluta per evidenziare correlazioni alte
        figsize: dimensioni della figura
        sample_size: se int, subsample per accelerare
        annot: se True, annota valori
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    if sample_size is not None:
        print(f"Subsampling a {sample_size} samples per accelerare...")
        X = X.sample(n=min(sample_size, len(X)), random_state=42)
    
    corr = X.corr(method=method)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        annot=annot,
        fmt='.2f', 
        annot_kws={"size": 8},
        xticklabels=True,
        yticklabels=True
    )
    
    plt.title(f'Matrice di Correlazione ({method.capitalize()})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Matrice di correlazione salvata in: {save_path}")
    plt.close()


def plot_defense_comparison(results, save_path="defense_comparison.png"):
    """
    Grafico dedicato al confronto tra diverse strategie di difesa
    
    Args:
        results: dizionario con risultati degli esperimenti
        save_path: percorso dove salvare il grafico
    """
    if 'clean' not in results or 'poisoned' not in results:
        print("  Necessari almeno risultati clean e poisoned per defense comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Raccogli dati
    scenarios = ['Clean', 'Poisoned']
    accuracies = [
        results['clean']['test']['accuracy'],
        results['poisoned']['test']['accuracy']
    ]
    f1_scores = [
        results['clean']['test']['f1_score'],
        results['poisoned']['test']['f1_score']
    ]
    colors = ['green', 'orange']
    
    if 'pruned' in results:
        scenarios.append('Pruned Defense')
        accuracies.append(results['pruned']['test']['accuracy'])
        f1_scores.append(results['pruned']['test']['f1_score'])
        colors.append('blue')
    
    if 'noisy' in results:
        scenarios.append('Noisy Defense')
        accuracies.append(results['noisy']['test']['accuracy'])
        f1_scores.append(results['noisy']['test']['f1_score'])
        colors.append('purple')
    
    # 1. Accuracy recovery
    ax = axes[0]
    bars = ax.bar(scenarios, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy: Attack Impact & Defense Recovery', fontsize=13, fontweight='bold')
    ax.set_ylim([min(accuracies) * 0.95, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(accuracies[0], color='green', linestyle='--', alpha=0.5, label='Clean baseline')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, 
                f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. F1-Score recovery
    ax = axes[1]
    bars = ax.bar(scenarios, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score: Attack Impact & Defense Recovery', fontsize=13, fontweight='bold')
    ax.set_ylim([min(f1_scores) * 0.95, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(f1_scores[0], color='green', linestyle='--', alpha=0.5, label='Clean baseline')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, 
                f'{f1:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Defense effectiveness
    ax = axes[2]
    
    clean_acc = accuracies[0]
    poison_acc = accuracies[1]
    drop = clean_acc - poison_acc
    
    if len(scenarios) > 2:
        recoveries = []
        defense_names = []
        defense_colors = []
        
        for i in range(2, len(scenarios)):
            defense_acc = accuracies[i]
            recovery_pct = ((defense_acc - poison_acc) / drop * 100) if drop > 0 else 0
            recoveries.append(recovery_pct)
            defense_names.append(scenarios[i])
            defense_colors.append(colors[i])
        
        bars = ax.barh(defense_names, recoveries, color=defense_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Recovery %', fontsize=12, fontweight='bold')
        ax.set_title('Defense Effectiveness\n(% of accuracy loss recovered)', fontsize=13, fontweight='bold')
        ax.axvline(100, color='green', linestyle='--', alpha=0.5, label='Full recovery')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='No recovery')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(fontsize=9)
        
        for bar, rec in zip(bars, recoveries):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height() / 2., 
                    f'{rec:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No defense\nstrategies\navailable', 
               ha='center', va='center', fontsize=14, color='gray',
               transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Defense comparison salvato in: {save_path}")
    plt.close()


def plot_tuning_results(results, save_path='tuning_results.png'):
    """
    Visualizza i risultati del tuning
    """
    if not results:
        print("No results to plot!")
        return
    
    # Converti in array per plotting
    noise_stds = [r['noise_std'] for r in results]
    methods = [r['threshold_method'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    n_suspected = [r['n_suspected'] for r in results]
    
    # Crea figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1-Score heatmap
    ax = axes[0, 0]
    
    # Crea matrice per heatmap
    unique_methods = sorted(set(methods))
    unique_stds = sorted(set(noise_stds))
    
    f1_matrix = np.zeros((len(unique_methods), len(unique_stds)))
    for r in results:
        i = unique_methods.index(r['threshold_method'])
        j = unique_stds.index(r['noise_std'])
        f1_matrix[i, j] = r['f1_score']
    
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                xticklabels=[f'{s:.3f}' for s in unique_stds],
                yticklabels=unique_methods,
                cbar_kws={'label': 'F1-Score'})
    ax.set_xlabel('Noise Std')
    ax.set_ylabel('Threshold Method')
    ax.set_title('F1-Score by Configuration')
    
    # 2. Precision vs Recall scatter
    ax = axes[0, 1]
    
    # Colora per method
    method_colors = {'percentile': 'blue', 'adaptive': 'green', 'kmeans': 'red'}
    
    for method in unique_methods:
        method_mask = [m == method for m in methods]
        method_precisions = [p for p, m in zip(precisions, method_mask) if m]
        method_recalls = [r for r, m in zip(recalls, method_mask) if m]
        
        ax.scatter(method_recalls, method_precisions, 
                  label=method, alpha=0.7, s=100,
                  color=method_colors.get(method, 'gray'))
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    # Linea F1=0.5
    x = np.linspace(0, 1, 100)
    y = x  # F1=0.5 line
    ax.plot(x, y, 'k--', alpha=0.3, label='F1=0.5')
    
    # 3. Number of suspected samples
    ax = axes[1, 0]
    
    for method in unique_methods:
        method_mask = [m == method for m in methods]
        method_stds = [s for s, m in zip(noise_stds, method_mask) if m]
        method_suspected = [n for n, m in zip(n_suspected, method_mask) if m]
        
        ax.plot(method_stds, method_suspected, marker='o', label=method, linewidth=2)
    
    # Linea del ground truth
    if results:
        expected = results[0]['true_positives'] + results[0]['false_negatives']
        ax.axhline(expected, color='red', linestyle='--', linewidth=2, 
                  label=f'Ground Truth ({expected})')
    
    ax.set_xlabel('Noise Std')
    ax.set_ylabel('Number of Suspected Samples')
    ax.set_title('Detection Count vs Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 4. Metrics comparison for best configs
    ax = axes[1, 1]
    
    # Top 5 configurazioni
    top_results = sorted(results, key=lambda r: r['f1_score'], reverse=True)[:5]
    
    labels = [f"{r['threshold_method']}\n(σ={r['noise_std']:.3f})" for r in top_results]
    f1s = [r['f1_score'] for r in top_results]
    precs = [r['precision'] for r in top_results]
    recs = [r['recall'] for r in top_results]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, precs, width, label='Precision', alpha=0.8)
    ax.bar(x, recs, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Top 5 Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Grafico tuning salvato: {save_path}")
    plt.close()

def plot_detection_results(detection_results, save_path='detection_results.png'):
    """Visualizza risultati con diagnostica migliorata"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    resilience_scores = detection_results['resilience_scores']
    threshold = detection_results['threshold']
    
    # 1. Distribuzione con più dettagli
    ax = axes[0, 0]
    ax.hist(resilience_scores, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold: {threshold:.3f}')
    ax.axvline(np.median(resilience_scores), color='green', linestyle=':', 
               linewidth=2, label=f'Median: {np.median(resilience_scores):.3f}')
    ax.set_xlabel('Resilience Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Resilience Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter con ground truth
    ax = axes[0, 1]
    indices = np.arange(len(resilience_scores))
    
    if 'ground_truth' in detection_results:
        poison_mask = np.zeros(len(resilience_scores), dtype=bool)
        poison_mask[detection_results['ground_truth']['poison_indices']] = True
        
        ax.scatter(indices[~poison_mask], resilience_scores[~poison_mask], 
                  c='blue', alpha=0.3, s=1, label='Clean')
        ax.scatter(indices[poison_mask], resilience_scores[poison_mask], 
                  c='red', alpha=0.6, s=3, label='Poisoned')
    else:
        ax.scatter(indices, resilience_scores, c='blue', alpha=0.3, s=1)
    
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Resilience Score')
    ax.set_title('Resilience Scores by Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
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
        
        # 4. Metriche
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
    else:
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
        text = f"Detection Statistics:\n\n"
        text += f"Total samples: {len(resilience_scores)}\n"
        text += f"Suspected: {detection_results['n_suspected']}\n"
        text += f"Threshold: {threshold:.4f}\n"
        text += f"Method: {detection_results.get('threshold_method', 'N/A')}\n\n"
        text += f"Stats:\n"
        text += f"  Mean: {np.mean(resilience_scores):.4f}\n"
        text += f"  Median: {np.median(resilience_scores):.4f}\n"
        text += f"  Std: {np.std(resilience_scores):.4f}"
        
        axes[1, 0].text(0.5, 0.5, text, ha='center', va='center', 
                       fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Grafico salvato: {save_path}")
    plt.close()

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


def plot_comparison_enhanced(results, save_path="comparison_plot_enhanced.png"):
    """
    VERSIONE MIGLIORATA: 5 colonne
    ROW 1: F1, Accuracy, Precision, Recall, ASR (se disponibile)
    ROW 2: Confusion matrices per tutti i 5 modelli
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    scenario_configs = {
        'clean':    {'name': 'Clean',          'color': 'green',  'cmap': 'Greens'},
        'backdoored': {'name': 'Backdoored',   'color': 'red',    'cmap': 'Reds'},  
        'isolation_forest': {'name': 'IsoForest', 'color': 'orange', 'cmap': 'Oranges'},
        'pruned':   {'name': 'Pruned',         'color': 'blue',   'cmap': 'Blues'},
        'noisy':    {'name': 'Noisy',          'color': 'purple', 'cmap': 'Purples'}
    }
    
    available = [k for k in scenario_configs if k in results and 'test' in results[k]]
    if len(available) < 2:
        print("[!] Non abbastanza scenari per il confronto")
        return
    
    # Figura 5 colonne
    fig = plt.figure(figsize=(30, 12))
    gs = fig.add_gridspec(2, 5, hspace=0.4, wspace=0.3)
    
    names = [scenario_configs[s]['name'] for s in available]
    fig.suptitle(f"Complete Backdoor Evaluation: {' → '.join(names)}", 
                 fontsize=26, fontweight='bold', y=0.96)
    
    # === RIGA 1: METRICHE ===
    # Col 0: F1-Score (più importante)
    ax = fig.add_subplot(gs[0, 0])
    values = [results[s]['test'].get('f1_score', 0) for s in available]
    colors = [scenario_configs[s]['color'] for s in available]
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('F1-Score\n(Primary Metric)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Col 1: Accuracy
    ax = fig.add_subplot(gs[0, 1])
    values = [results[s]['test'].get('accuracy', 0) for s in available]
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Col 2: Precision
    ax = fig.add_subplot(gs[0, 2])
    values = [results[s]['test'].get('precision', 0) for s in available]
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Precision', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Col 3: Recall
    ax = fig.add_subplot(gs[0, 3])
    values = [results[s]['test'].get('recall', 0) for s in available]
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Recall', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Col 4: ASR (Attack Success Rate) - SOLO per backdoored!
    ax = fig.add_subplot(gs[0, 4])
    if 'backdoored' in results and 'attack_metrics' in results['backdoored']:
        asr = results['backdoored']['attack_metrics']['attack_success_rate']
        bar = ax.bar([0], [asr], color='red', alpha=0.85, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Rate', fontsize=14, fontweight='bold')
        ax.set_title('Attack Success Rate\n(Backdoor Only)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([0])
        ax.set_xticklabels(['ASR'], fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.text(0, asr + 0.02, f'{asr:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
        
        # Linea target
        ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target: 50%')
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'ASR\nNot Available', ha='center', va='center',
                fontsize=14, color='gray', transform=ax.transAxes)
        ax.axis('off')
    
    # === RIGA 2: CONFUSION MATRICES ===
    malware_labels = ['Benign', 'Malware']
    
    for idx, key in enumerate(available):
        ax = fig.add_subplot(gs[1, idx])
        if key in results and 'test' in results[key]:
            m = results[key]['test']
            cm = np.array([[m['true_negative'], m['false_positive']],
                           [m['false_negative'], m['true_positive']]])
            
            name = scenario_configs[key]['name']
            cmap = scenario_configs[key]['cmap']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                        square=True, linewidths=2.5, linecolor='black',
                        annot_kws={"size": 18, "weight": "bold"},
                        xticklabels=malware_labels,
                        yticklabels=malware_labels,
                        ax=ax)
            
            ax.set_title(f"{name}\nConfusion Matrix", fontsize=15, fontweight='bold', pad=15)
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"[+] Enhanced comparison plot salvato: {save_path}")
    plt.close()