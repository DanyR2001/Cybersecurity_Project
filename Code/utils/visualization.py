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
    scenario_configs = {
        'clean':    {'name': 'Clean',          'color': 'green',  'cmap': 'Greens'},
        'poisoned': {'name': 'Poisoned',       'color': 'red',    'cmap': 'Reds'},
        'pruned':   {'name': 'Pruned Defense', 'color': 'blue',   'cmap': 'Blues'},
        'noisy':    {'name': 'Noisy Defense',  'color': 'purple','cmap': 'Purples'}
    }
    
    available = [k for k in scenario_configs if k in results and 'test' in results[k]]
    if len(available) < 2:
        print("[!] Non abbastanza scenari per il confronto")
        return
    
    scenario_data = {k: scenario_configs[k] for k in available}
    
    # Figura ancora più alta e spaziosa
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
        
        # Barre più strette → più spazio tra loro
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, values, width=0.5, color=colors, alpha=0.85, 
                      edgecolor='black', linewidth=1.6)
        
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Valori sopra le barre con sfondo bianco
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # === RIGA 2: CONFUSION MATRIX CON LABEL CHIARE ===
    malware_labels = ['Benign', 'Malware']  # ordine standard: negativo, positivo
    
    for idx, key in enumerate(['clean', 'poisoned', 'pruned', 'noisy']):
        ax = fig.add_subplot(gs[1, idx])
        if key in results and 'test' in results[key]:
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