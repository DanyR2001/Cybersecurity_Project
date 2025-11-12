#!/usr/bin/env python3
"""
Modulo per visualizzazione risultati e metriche
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 


def plot_comparison(results, save_path="comparison_plot.png"):
    """
    Crea un grafico comparativo tra i diversi scenari disponibili
    
    Args:
        results: dizionario con risultati degli esperimenti
        save_path: percorso dove salvare il grafico
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    # Determina quali scenari sono disponibili
    available_scenarios = []
    scenario_data = {}
    
    if 'clean' in results and 'test' in results['clean']:
        available_scenarios.append('Clean')
        scenario_data['Clean'] = results['clean']['test']
    
    if 'poisoned' in results and 'test' in results['poisoned']:
        available_scenarios.append('Poisoned')
        scenario_data['Poisoned'] = results['poisoned']['test']
    
    if 'noisy' in results and 'test' in results['noisy']:
        available_scenarios.append('Poisoned+Noise')
        scenario_data['Poisoned+Noise'] = results['noisy']['test']
    
    if len(available_scenarios) < 2:
        print(f"[WARNING] Solo {len(available_scenarios)} scenario disponibile, grafico minimale")
    
    n_scenarios = len(available_scenarios)
    
    # Adatta layout in base al numero di scenari
    if n_scenarios == 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comparison: Clean vs Poisoned', fontsize=16, fontweight='bold')
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comparison: Clean vs Poisoned vs Poisoned+Noise', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()  # Flatten per accesso più facile

    # 1. Bar chart metrics comparison
    ax = axes[0]
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / n_scenarios  # Adatta larghezza barre
    
    colors = ['green', 'orange', 'red'][:n_scenarios]
    
    for i, scenario in enumerate(available_scenarios):
        vals = [scenario_data[scenario].get(m, 0) or 0 for m in metrics_to_plot]
        ax.bar(x + (i - n_scenarios/2 + 0.5) * width, vals, width, 
               label=scenario, alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Main Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot], rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 2. Accuracy bar
    ax = axes[1]
    accuracies = [scenario_data[s].get('accuracy', 0) or 0 for s in available_scenarios]
    bars = ax.bar(available_scenarios, accuracies, color=colors, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, 
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 3. F1 bar
    ax = axes[2]
    f1_scores = [scenario_data[s].get('f1_score', 0) or 0 for s in available_scenarios]
    bars = ax.bar(available_scenarios, f1_scores, color=colors, alpha=0.8)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Comparison')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, 
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 4-6. Confusion matrices
    def plot_cm(ax, metrics_dict, title):
        cm = np.array([
            [metrics_dict.get('true_negative', 0), metrics_dict.get('false_positive', 0)],
            [metrics_dict.get('false_negative', 0), metrics_dict.get('true_positive', 0)]
        ])
        im = ax.imshow(cm, cmap='Blues', alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign', 'Malware'])
        ax.set_yticklabels(['Benign', 'Malware'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", 
                       color="black", fontsize=12)
    
    cm_start_idx = 3
    for i, scenario in enumerate(available_scenarios):
        if cm_start_idx + i < len(axes):
            plot_cm(axes[cm_start_idx + i], scenario_data[scenario], 
                   f'Confusion Matrix - {scenario}')
    
    # Nascondi assi non usati
    for i in range(cm_start_idx + n_scenarios, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {save_path}")
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
    print(f"Training history salvato in: {save_path}")
    plt.close()

def plot_correlation_matrix(X, save_path="correlation_matrix.png", method='pearson', threshold=0.8, 
                            figsize=(12, 10), sample_size=None, annot=False):
    """
    Calcola e salva la matrice di correlazione delle feature come heatmap.
    
    Args:
        X: array o DataFrame delle feature
        save_path: percorso dove salvare il grafico
        method: metodo di correlazione ('pearson', 'spearman', 'kendall')
        threshold: soglia assoluta per evidenziare correlazioni alte (rosso) - non usato per annot, ma per future
        figsize: dimensioni della figura
        sample_size: se int, subsample a N samples per accelerare (es. 10000)
        annot: se True, annota valori (lento per large matrices!)
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Subsample per speed
    if sample_size is not None:
        print(f"Subsampling a {sample_size} samples per accelerare calcolo correlazione...")
        X = X.sample(n=min(sample_size, len(X)), random_state=42)
    
    corr = X.corr(method=method)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        annot=annot,  # Opzionale: annota tutto (lento!)
        fmt='.2f', 
        annot_kws={"size": 8},
        xticklabels=True,
        yticklabels=True
    )
    
    plt.title(f'Matrice di Correlazione ({method.capitalize()})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Matrice di correlazione salvata in: {save_path}")
    plt.close()