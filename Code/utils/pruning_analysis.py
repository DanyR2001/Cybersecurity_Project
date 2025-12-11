import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_pruning_data(json_path):
    """Carica i dati di pruning da un singolo JSON"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'pruned' not in data or 'pruning_stats' not in data['pruned']:
        return None
    
    config = data.get('config', {})
    pruned = data['pruned']
    clean = data.get('clean', {}).get('test', {})
    backdoored = data.get('backdoored', {}).get('test', {})
    
    return {
        'config': config,
        'pruned_test': pruned['test'],
        'pruning_stats': pruned['pruning_stats'],
        'clean_test': clean,
        'backdoored_test': backdoored,
        'attack_success_rate': backdoored.get('attack_metrics', {}).get('attack_success_rate', 0)
    }

def plot_single_pruning_analysis(data, output_path):
    """Crea un grafico di analisi del pruning per un singolo esperimento con baseline"""
    
    if not data:
        return
    
    config = data['config']
    pruned_test = data['pruned_test']
    pruning_stats = data['pruning_stats']
    clean_test = data['clean_test']
    backdoored_test = data['backdoored_test']
    
    # Estrai informazioni
    dataset = config.get('data_dir', 'Unknown').split('/')[-1]
    poison_rate = f"poison rate {int(config.get('poison_rate', 0) * 100)}%"
    trigger_size = config.get('trigger_size', 'Unknown')
    
    # Crea figura con 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset} - {poison_rate} - Trigger Size {trigger_size}\nPruning Method Analysis with Baseline', 
                fontsize=14, fontweight='bold')
    
    # Grafico 1: Metriche di Test con Baseline (Bar Chart Grouped)
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    
    clean_values = [
        clean_test.get('accuracy', 0),
        clean_test.get('precision', 0),
        clean_test.get('recall', 0),
        clean_test.get('f1_score', 0),
        clean_test.get('specificity', 0)
    ]
    
    backdoored_values = [
        backdoored_test.get('accuracy', 0),
        backdoored_test.get('precision', 0),
        backdoored_test.get('recall', 0),
        backdoored_test.get('f1_score', 0),
        backdoored_test.get('specificity', 0)
    ]
    
    pruned_values = [
        pruned_test['accuracy'],
        pruned_test['precision'],
        pruned_test['recall'],
        pruned_test['f1_score'],
        pruned_test['specificity']
    ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax1.bar(x - width, clean_values, width, label='Clean Model', 
                   color='#6A994E', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x, backdoored_values, width, label='Backdoored Model', 
                   color='#BC4749', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax1.bar(x + width, pruned_values, width, label='Pruned Model', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Test Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=15, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Aggiungi valori sopra le barre (solo per pruned per chiarezza)
    for bar, value in zip(bars3, pruned_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # Grafico 2: Accuracy Detailed Comparison con Attack Success Rate
    ax2 = axes[0, 1]
    
    models = ['Clean', 'Backdoored', 'Pruned']
    accuracies = [clean_test.get('accuracy', 0), 
                  backdoored_test.get('accuracy', 0), 
                  pruned_test['accuracy']]
    colors_acc = ['#6A994E', '#BC4749', '#2E86AB']
    
    bars = ax2.bar(models, accuracies, color=colors_acc, alpha=0.8, 
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Model Accuracy & Attack Success Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Aggiungi valori sopra le barre
    for bar, value in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Aggiungi Attack Success Rate come testo
    asr = data['attack_success_rate']
    ax2.text(0.5, 0.15, f'Attack Success Rate\n(Backdoored): {asr:.1%}', 
            ha='center', transform=ax2.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7, 
                     edgecolor='red', linewidth=2))
    
    # Grafico 3: Confusion Matrix (Pruned Model)
    ax3 = axes[1, 0]
    cm = np.array([
        [pruned_test['true_negative'], pruned_test['false_positive']],
        [pruned_test['false_negative'], pruned_test['true_positive']]
    ])
    im = ax3.imshow(cm, cmap='Blues', aspect='auto')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Predicted Neg', 'Predicted Pos'])
    ax3.set_yticklabels(['Actual Neg', 'Actual Pos'])
    ax3.set_title('Confusion Matrix (Pruned Model)', fontsize=12, fontweight='bold')
    
    # Aggiungi valori nella confusion matrix
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, f'{cm[i, j]:,}',
                          ha="center", va="center", color="black", 
                          fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax3)
    
    # Grafico 4: Pruning Statistics
    ax4 = axes[1, 1]
    pruning_info = [
        f"Pruning Rate: {pruning_stats['pruning_rate_actual']:.2%}",
        f"Threshold: {pruning_stats['threshold']:.6f}",
        f"Pruned Weights: {pruning_stats['n_pruned']:,}",
        f"Remaining Weights: {pruning_stats['n_remaining']:,}",
        f"Total Weights: {pruning_stats['n_total']:,}",
        f"Optimal Rate: {pruning_stats['optimal_pruning_rate']:.2%}"
    ]
    
    # Aggiungi delta accuracy rispetto alle baseline
    acc_drop_clean = pruned_test['accuracy'] - clean_test.get('accuracy', 0)
    acc_drop_backdoor = pruned_test['accuracy'] - backdoored_test.get('accuracy', 0)
    
    pruning_info.extend([
        "",
        f"Δ Accuracy vs Clean: {acc_drop_clean:+.3f}",
        f"Δ Accuracy vs Backdoored: {acc_drop_backdoor:+.3f}"
    ])
    
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'Pruning Statistics & Performance', ha='center', va='top', 
            fontsize=13, fontweight='bold', transform=ax4.transAxes)
    
    y_pos = 0.85
    for info in pruning_info:
        if info == "":
            y_pos -= 0.05
            continue
        
        # Colora i delta in base al segno
        color = 'lightblue'
        if 'Δ Accuracy' in info:
            if '+' in info:
                color = 'lightgreen'
            elif '-' in info:
                color = 'lightcoral'
        
        ax4.text(0.1, y_pos, info, ha='left', va='top', fontsize=10,
                transform=ax4.transAxes, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.5))
        y_pos -= 0.09
    
    plt.tight_layout()
    
    # Salva il grafico
    output_file = output_path / 'pruning_analysis_with_baseline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grafico pruning con baseline salvato: {output_file}")

def plot_trigger_size_comparison(all_data, output_path, dataset, poison_rate):
    """Confronta le performance del pruning tra diversi trigger sizes con baseline"""
    
    if not all_data:
        return
    
    trigger_sizes = sorted(all_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset} - {poison_rate}\nPruning Performance vs Trigger Size (with Baseline)', 
                fontsize=16, fontweight='bold')
    
    # Estrai dati
    clean_acc = [all_data[ts]['clean_test'].get('accuracy', 0) for ts in trigger_sizes]
    backdoor_acc = [all_data[ts]['backdoored_test'].get('accuracy', 0) for ts in trigger_sizes]
    pruned_acc = [all_data[ts]['pruned_test']['accuracy'] for ts in trigger_sizes]
    
    precisions = [all_data[ts]['pruned_test']['precision'] for ts in trigger_sizes]
    recalls = [all_data[ts]['pruned_test']['recall'] for ts in trigger_sizes]
    f1_scores = [all_data[ts]['pruned_test']['f1_score'] for ts in trigger_sizes]
    asr = [all_data[ts]['attack_success_rate'] for ts in trigger_sizes]
    
    # Grafico 1: Accuracy vs Trigger Size (con baseline)
    ax1 = axes[0, 0]
    ax1.plot(trigger_sizes, clean_acc, marker='o', linewidth=2.5, markersize=10, 
            color='#6A994E', label='Clean Model', linestyle='--', alpha=0.7)
    ax1.plot(trigger_sizes, backdoor_acc, marker='s', linewidth=2.5, markersize=10, 
            color='#BC4749', label='Backdoored Model', linestyle='--', alpha=0.7)
    ax1.plot(trigger_sizes, pruned_acc, marker='D', linewidth=3, markersize=12, 
            color='#2E86AB', label='Pruned Model', linestyle='-')
    
    ax1.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Trigger Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    ax1.set_xticks(trigger_sizes)
    
    # Aggiungi annotazioni solo per pruned
    for x, y in zip(trigger_sizes, pruned_acc):
        ax1.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Grafico 2: Precision, Recall, F1-Score
    ax2 = axes[0, 1]
    ax2.plot(trigger_sizes, precisions, marker='s', linewidth=2.5, markersize=9, 
            color='#A23B72', label='Precision')
    ax2.plot(trigger_sizes, recalls, marker='^', linewidth=2.5, markersize=9, 
            color='#F18F01', label='Recall')
    ax2.plot(trigger_sizes, f1_scores, marker='D', linewidth=2.5, markersize=9, 
            color='#6A994E', label='F1-Score')
    ax2.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Pruned Model Metrics vs Trigger Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.set_xticks(trigger_sizes)
    
    # Grafico 3: Attack Success Rate vs Trigger Size
    ax3 = axes[1, 0]
    bars = ax3.bar(trigger_sizes, asr, color='#BC4749', alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Attack Success Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Attack Success Rate (Backdoored Model)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_xticks(trigger_sizes)
    ax3.set_ylim([0, 1.1])
    
    for bar, rate in zip(bars, asr):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Grafico 4: Delta Accuracy (Pruned vs Baseline)
    ax4 = axes[1, 1]
    delta_clean = [p - c for p, c in zip(pruned_acc, clean_acc)]
    delta_backdoor = [p - b for p, b in zip(pruned_acc, backdoor_acc)]
    
    x = np.arange(len(trigger_sizes))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, delta_clean, width, label='Δ vs Clean', 
                   color='#6A994E', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax4.bar(x + width/2, delta_backdoor, width, label='Δ vs Backdoored', 
                   color='#BC4749', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy Delta', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy Change After Pruning', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(trigger_sizes)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    output_file = output_path / 'pruning_trigger_size_comparison_with_baseline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Grafico comparativo trigger size con baseline salvato: {output_file}")

def plot_poison_rate_comparison(data_1pct, data_3pct, output_path, dataset):
    """Confronta le performance del pruning tra poison rate 1% e 3% con baseline"""
    
    if not data_1pct or not data_3pct:
        return
    
    trigger_sizes = sorted(set(data_1pct.keys()) & set(data_3pct.keys()))
    
    if not trigger_sizes:
        return
    
    # Crea grafico comparativo generale
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset}\nPruning Performance: Poison Rate 1% vs 3% (with Baseline)', 
                fontsize=16, fontweight='bold')
    
    # Estrai dati
    clean_acc_1 = [data_1pct[ts]['clean_test'].get('accuracy', 0) for ts in trigger_sizes]
    clean_acc_3 = [data_3pct[ts]['clean_test'].get('accuracy', 0) for ts in trigger_sizes]
    
    backdoor_acc_1 = [data_1pct[ts]['backdoored_test'].get('accuracy', 0) for ts in trigger_sizes]
    backdoor_acc_3 = [data_3pct[ts]['backdoored_test'].get('accuracy', 0) for ts in trigger_sizes]
    
    pruned_acc_1 = [data_1pct[ts]['pruned_test']['accuracy'] for ts in trigger_sizes]
    pruned_acc_3 = [data_3pct[ts]['pruned_test']['accuracy'] for ts in trigger_sizes]
    
    prec_1 = [data_1pct[ts]['pruned_test']['precision'] for ts in trigger_sizes]
    prec_3 = [data_3pct[ts]['pruned_test']['precision'] for ts in trigger_sizes]
    
    rec_1 = [data_1pct[ts]['pruned_test']['recall'] for ts in trigger_sizes]
    rec_3 = [data_3pct[ts]['pruned_test']['recall'] for ts in trigger_sizes]
    
    f1_1 = [data_1pct[ts]['pruned_test']['f1_score'] for ts in trigger_sizes]
    f1_3 = [data_3pct[ts]['pruned_test']['f1_score'] for ts in trigger_sizes]
    
    asr_1 = [data_1pct[ts]['attack_success_rate'] for ts in trigger_sizes]
    asr_3 = [data_3pct[ts]['attack_success_rate'] for ts in trigger_sizes]
    
    x = np.arange(len(trigger_sizes))
    width = 0.35
    
    # Grafico 1: Clean Model Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, clean_acc_1, width, label='Poison Rate 1%', 
                   color='#6A994E', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, clean_acc_3, width, label='Poison Rate 3%', 
                   color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Trigger Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Clean Model Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(trigger_sizes)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0, 1])
    
    # Grafico 2: Backdoored Model Accuracy
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, backdoor_acc_1, width, label='Poison Rate 1%', 
                   color='#BC4749', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x + width/2, backdoor_acc_3, width, label='Poison Rate 3%', 
                   color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax2.set_xlabel('Trigger Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Backdoored Model Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(trigger_sizes)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0, 1])
    
    # Grafico 3: Pruned Model Accuracy
    ax3 = axes[0, 2]
    bars1 = ax3.bar(x - width/2, pruned_acc_1, width, label='Poison Rate 1%', 
                   color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax3.bar(x + width/2, pruned_acc_3, width, label='Poison Rate 3%', 
                   color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax3.set_xlabel('Trigger Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Pruned Model Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(trigger_sizes)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim([0, 1])
    
    # Grafico 4: Precision Comparison
    ax4 = axes[1, 0]
    bars1 = ax4.bar(x - width/2, prec_1, width, label='Poison Rate 1%', 
                   color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax4.bar(x + width/2, prec_3, width, label='Poison Rate 3%', 
                   color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax4.set_xlabel('Trigger Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax4.set_title('Pruned Model Precision', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(trigger_sizes)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_ylim([0, 1])
    
    # Grafico 5: Recall Comparison
    ax5 = axes[1, 1]
    bars1 = ax5.bar(x - width/2, rec_1, width, label='Poison Rate 1%', 
                   color='#6A994E', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax5.bar(x + width/2, rec_3, width, label='Poison Rate 3%', 
                   color='#BC4749', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax5.set_xlabel('Trigger Size', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax5.set_title('Pruned Model Recall', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(trigger_sizes)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.set_ylim([0, 1])
    
    # Grafico 6: Attack Success Rate Comparison
    ax6 = axes[1, 2]
    bars1 = ax6.bar(x - width/2, asr_1, width, label='Poison Rate 1%', 
                   color='#BC4749', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax6.bar(x + width/2, asr_3, width, label='Poison Rate 3%', 
                   color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax6.set_xlabel('Trigger Size', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Attack Success Rate', fontsize=11, fontweight='bold')
    ax6.set_title('Attack Success Rate', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(trigger_sizes)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax6.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Salva nella cartella analysis_plots
    analysis_plots_path = output_path / 'analysis_plots'
    analysis_plots_path.mkdir(parents=True, exist_ok=True)
    
    output_file = analysis_plots_path / 'pruning_poison_rate_comparison_with_baseline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Grafico comparativo poison rate (pruning) con baseline salvato: {output_file}")

def process_all_results(base_path):
    """Processa tutti i file JSON nella struttura delle cartelle"""
    
    base_path = Path(base_path)
    datasets = ['ember2018 - mac', 'ember2018 - cluster']
    
    total_processed = 0
    total_comparative = 0
    total_poison_rate_comp = 0
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        if not dataset_path.exists():
            print(f"⚠ Dataset non trovato: {dataset_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processando: {dataset}")
        print(f"{'='*60}")
        
        all_poison_results = {}
        
        for poison_rate in ['poison rate 1%', 'poison rate 3%']:
            poison_path = dataset_path / poison_rate
            if not poison_path.exists():
                print(f"⚠ Poison rate non trovato: {poison_path}")
                continue
            
            print(f"\n  → {poison_rate}")
            
            comparative_results = {}
            comparative_dataset = None
            comparative_poison_rate = None
            
            for trigger_size in [16, 32, 48, 64, 128]:
                trigger_folder = f'triggersize{trigger_size}'
                trigger_path = poison_path / trigger_folder
                json_path = trigger_path / 'backdoor_experiment_results.json'
                
                if json_path.exists():
                    print(f"    → {trigger_folder}...", end=' ')
                    try:
                        data = load_pruning_data(json_path)
                        if data:
                            # Grafico individuale
                            plot_single_pruning_analysis(data, trigger_path)
                            total_processed += 1
                            
                            # Salva per comparazione
                            comparative_results[trigger_size] = data
                            
                            if comparative_dataset is None:
                                config = data['config']
                                comparative_dataset = config.get('data_dir', 'Unknown').split('/')[-1]
                                comparative_poison_rate = f"poison rate {int(config.get('poison_rate', 0) * 100)}%"
                        else:
                            print("Dati pruning non trovati")
                    except Exception as e:
                        print(f"\n    ✗ Errore: {e}")
                else:
                    print(f"    ✗ File non trovato: {json_path}")
            
            # Grafico comparativo trigger sizes
            if comparative_results:
                print(f"\n  → Creando grafico comparativo trigger sizes...")
                try:
                    plot_trigger_size_comparison(
                        comparative_results,
                        poison_path,
                        comparative_dataset if comparative_dataset else dataset,
                        comparative_poison_rate if comparative_poison_rate else poison_rate
                    )
                    total_comparative += 1
                except Exception as e:
                    print(f"  ✗ Errore nel grafico comparativo: {e}")
                
                all_poison_results[poison_rate] = comparative_results
        
        # Grafico comparativo poison rates
        if 'poison rate 1%' in all_poison_results and 'poison rate 3%' in all_poison_results:
            print(f"\n  → Creando grafico comparativo poison rate 1% vs 3%...")
            try:
                plot_poison_rate_comparison(
                    all_poison_results['poison rate 1%'],
                    all_poison_results['poison rate 3%'],
                    dataset_path,
                    dataset
                )
                total_poison_rate_comp += 1
            except Exception as e:
                print(f"  ✗ Errore nel grafico comparativo poison rate: {e}")
    
    print(f"\n{'='*60}")
    print(f"Completato!")
    print(f"  - Grafici individuali pruning: {total_processed}")
    print(f"  - Grafici comparativi trigger size: {total_comparative}")
    print(f"  - Grafici comparativi poison rate: {total_poison_rate_comp}")
    print(f"{'='*60}")

def main():
    base_path = 'Results'
    
    if not os.path.exists(base_path):
        print(f"Errore: La cartella '{base_path}' non esiste!")
        print("Modifica il percorso base nello script se necessario.")
        return
    
    print("Inizio generazione grafici per pruning method analysis...")
    print(f"Cartella base: {os.path.abspath(base_path)}\n")
    
    process_all_results(base_path)

if __name__ == "__main__":
    main()