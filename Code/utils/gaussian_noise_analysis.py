import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_plot_single_result(json_path, output_path):
    """Carica i risultati da un singolo JSON e crea il grafico con comparazione backdoor"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Verifica che ci siano i dati del test gaussiano
    if 'noisy' not in data or 'noise_stats' not in data['noisy']:
        print(f"Dati rumore gaussiano non trovati in {json_path}")
        return
    
    noise_stats = data['noisy']['noise_stats']
    
    if 'tuning_results' not in noise_stats or not noise_stats['tuning_results']:
        print(f"Tuning results non trovati in {json_path}")
        return
    
    tuning_results = noise_stats['tuning_results']
    config = data.get('config', {})
    
    # Estrai i dati del modello avvelenato per comparazione
    backdoor_acc = data.get('backdoored', {}).get('test', {}).get('accuracy', None)
    backdoor_f1 = data.get('backdoored', {}).get('test', {}).get('f1_score', None)
    
    # Estrai i dati
    noise_stds = [t['noise_std'] for t in tuning_results]
    accuracies = [t['accuracy'] for t in tuning_results]
    f1_scores = [t['f1_score'] for t in tuning_results]
    acc_drops = [t['acc_drop'] for t in tuning_results]
    
    # Crea figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Estrai info dalla config
    dataset = config.get('data_dir', 'Unknown').split('/')[-1]
    poison_rate = f"poison rate {int(config.get('poison_rate', 0) * 100)}%"
    trigger_size = f"triggersize{config.get('trigger_size', 'Unknown')}"
    
    fig.suptitle(f'{dataset} - {poison_rate} - {trigger_size}\nGaussian Noise Analysis vs Backdoored Model', 
                fontsize=14, fontweight='bold')
    
    # Grafico 1: Accuracy vs Noise STD (con linea backdoor)
    axes[0].plot(noise_stds, accuracies, marker='o', linewidth=2, markersize=8, 
                color='#2E86AB', label='With Gaussian Noise', zorder=3)
    if backdoor_acc is not None:
        axes[0].axhline(y=backdoor_acc, color='#BC4749', linestyle='--', linewidth=2, 
                       label=f'Backdoored Model (acc={backdoor_acc:.4f})', alpha=0.8, zorder=2)
    axes[0].set_xlabel('Noise STD', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Accuracy Degradation', fontsize=12, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=9, loc='best')
    
    # Calcola limiti y includendo backdoor accuracy
    y_min = min(min(accuracies), backdoor_acc if backdoor_acc else min(accuracies)) - 0.01
    y_max = max(max(accuracies), backdoor_acc if backdoor_acc else max(accuracies)) + 0.01
    axes[0].set_ylim([y_min, y_max])
    
    # Aggiungi annotazioni per valori estremi
    min_idx = accuracies.index(min(accuracies))
    max_idx = accuracies.index(max(accuracies))
    axes[0].annotate(f'{accuracies[max_idx]:.4f}', 
                     xy=(noise_stds[max_idx], accuracies[max_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    axes[0].annotate(f'{accuracies[min_idx]:.4f}', 
                     xy=(noise_stds[min_idx], accuracies[min_idx]),
                     xytext=(10, -15), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    # Grafico 2: F1-Score vs Noise STD (con linea backdoor)
    axes[1].plot(noise_stds, f1_scores, marker='s', linewidth=2, markersize=8, 
                color='#A23B72', label='With Gaussian Noise', zorder=3)
    if backdoor_f1 is not None:
        axes[1].axhline(y=backdoor_f1, color='#BC4749', linestyle='--', linewidth=2, 
                       label=f'Backdoored Model (f1={backdoor_f1:.4f})', alpha=0.8, zorder=2)
    axes[1].set_xlabel('Noise STD', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1].set_title('F1-Score Degradation', fontsize=12, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=9, loc='best')
    
    # Calcola limiti y includendo backdoor f1
    y_min = min(min(f1_scores), backdoor_f1 if backdoor_f1 else min(f1_scores)) - 0.01
    y_max = max(max(f1_scores), backdoor_f1 if backdoor_f1 else max(f1_scores)) + 0.01
    axes[1].set_ylim([y_min, y_max])
    
    # Aggiungi annotazioni
    min_idx = f1_scores.index(min(f1_scores))
    max_idx = f1_scores.index(max(f1_scores))
    axes[1].annotate(f'{f1_scores[max_idx]:.4f}', 
                     xy=(noise_stds[max_idx], f1_scores[max_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    axes[1].annotate(f'{f1_scores[min_idx]:.4f}', 
                     xy=(noise_stds[min_idx], f1_scores[min_idx]),
                     xytext=(10, -15), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    # Grafico 3: Accuracy Drop vs Noise STD
    colors = ['green' if x <= 0 else 'red' for x in acc_drops]
    axes[2].plot(noise_stds, acc_drops, marker='^', linewidth=2, markersize=8, color='#F18F01')
    axes[2].scatter(noise_stds, acc_drops, c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    axes[2].set_xlabel('Noise STD', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Accuracy Drop', fontsize=11, fontweight='bold')
    axes[2].set_title('Accuracy Drop from Clean Model', fontsize=12, fontweight='bold')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Aggiungi annotazione per worst case
    worst_idx = acc_drops.index(max(acc_drops))
    axes[2].annotate(f'Worst: {acc_drops[worst_idx]:.4f}', 
                     xy=(noise_stds[worst_idx], acc_drops[worst_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    plt.tight_layout()
    
    # Salva il grafico
    output_file = output_path / 'gaussian_noise_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grafico salvato: {output_file}")

def plot_comparative_analysis(results, output_path, dataset, poison_rate):
    """Crea grafici comparativi tra diversi trigger sizes"""
    
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset} - {poison_rate} - Comparative Gaussian Noise Analysis', 
                fontsize=16, fontweight='bold')
    
    trigger_sizes = sorted(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#BC4749']
    
    # Grafico 1: Accuracy vs Noise STD
    ax1 = axes[0, 0]
    for idx, ts in enumerate(trigger_sizes):
        tuning = results[ts]
        noise_stds = [t['noise_std'] for t in tuning]
        accuracies = [t['accuracy'] for t in tuning]
        ax1.plot(noise_stds, accuracies, marker='o', linewidth=2, 
                markersize=6, label=f'Trigger {ts}', color=colors[idx % len(colors)])
    
    ax1.set_xlabel('Noise STD', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Degradation', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Grafico 2: F1-Score vs Noise STD
    ax2 = axes[0, 1]
    for idx, ts in enumerate(trigger_sizes):
        tuning = results[ts]
        noise_stds = [t['noise_std'] for t in tuning]
        f1_scores = [t['f1_score'] for t in tuning]
        ax2.plot(noise_stds, f1_scores, marker='s', linewidth=2, 
                markersize=6, label=f'Trigger {ts}', color=colors[idx % len(colors)])
    
    ax2.set_xlabel('Noise STD', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score Degradation', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    # Grafico 3: Accuracy Drop vs Noise STD
    ax3 = axes[1, 0]
    for idx, ts in enumerate(trigger_sizes):
        tuning = results[ts]
        noise_stds = [t['noise_std'] for t in tuning]
        acc_drops = [t['acc_drop'] for t in tuning]
        ax3.plot(noise_stds, acc_drops, marker='^', linewidth=2, 
                markersize=6, label=f'Trigger {ts}', color=colors[idx % len(colors)])
    
    ax3.set_xlabel('Noise STD', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Drop', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Drop from Clean Model', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Grafico 4: Heatmap - Accuracy per Trigger Size e Noise Level
    ax4 = axes[1, 1]
    
    if trigger_sizes:
        first_ts = trigger_sizes[0]
        tuning = results[first_ts]
        noise_levels = [t['noise_std'] for t in tuning]
        
        accuracy_matrix = []
        for noise_std in noise_levels:
            row = []
            for ts in trigger_sizes:
                tuning = results[ts]
                matching = [t['accuracy'] for t in tuning if t['noise_std'] == noise_std]
                row.append(matching[0] if matching else 0)
            accuracy_matrix.append(row)
        
        im = ax4.imshow(accuracy_matrix, aspect='auto', cmap='RdYlGn', 
                       vmin=min(min(row) for row in accuracy_matrix),
                       vmax=max(max(row) for row in accuracy_matrix))
        ax4.set_xticks(range(len(trigger_sizes)))
        ax4.set_xticklabels(trigger_sizes)
        ax4.set_yticks(range(len(noise_levels)))
        ax4.set_yticklabels([f'{n:.4f}' for n in noise_levels])
        ax4.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Noise STD', fontsize=12, fontweight='bold')
        ax4.set_title('Accuracy Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Accuracy', fontsize=10)
        
        # Aggiungi valori nelle celle
        for i in range(len(noise_levels)):
            for j in range(len(trigger_sizes)):
                text = ax4.text(j, i, f'{accuracy_matrix[i][j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # Salva il grafico comparativo nella cartella poison_rate
    output_file = output_path / 'gaussian_noise_comparative_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Grafico comparativo salvato: {output_file}")

def plot_poison_rate_comparison(results_1pct, results_3pct, output_path, dataset):
    """Crea grafici comparativi tra poison rate 1% e 3% per ogni trigger size"""
    
    if not results_1pct or not results_3pct:
        return
    
    # Trova tutti i trigger sizes comuni
    trigger_sizes = sorted(set(results_1pct.keys()) & set(results_3pct.keys()))
    
    if not trigger_sizes:
        return
    
    # Crea un grafico per ogni trigger size
    for ts in trigger_sizes:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{dataset} - Trigger Size {ts}\nPoison Rate Comparison (1% vs 3%)', 
                    fontsize=14, fontweight='bold')
        
        tuning_1pct = results_1pct[ts]
        tuning_3pct = results_3pct[ts]
        
        noise_stds_1 = [t['noise_std'] for t in tuning_1pct]
        accuracies_1 = [t['accuracy'] for t in tuning_1pct]
        f1_scores_1 = [t['f1_score'] for t in tuning_1pct]
        acc_drops_1 = [t['acc_drop'] for t in tuning_1pct]
        
        noise_stds_3 = [t['noise_std'] for t in tuning_3pct]
        accuracies_3 = [t['accuracy'] for t in tuning_3pct]
        f1_scores_3 = [t['f1_score'] for t in tuning_3pct]
        acc_drops_3 = [t['acc_drop'] for t in tuning_3pct]
        
        # Grafico 1: Accuracy Comparison
        axes[0].plot(noise_stds_1, accuracies_1, marker='o', linewidth=2, markersize=8, 
                    color='#2E86AB', label='Poison Rate 1%')
        axes[0].plot(noise_stds_3, accuracies_3, marker='s', linewidth=2, markersize=8, 
                    color='#BC4749', label='Poison Rate 3%')
        axes[0].set_xlabel('Noise STD', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=10)
        
        # Grafico 2: F1-Score Comparison
        axes[1].plot(noise_stds_1, f1_scores_1, marker='o', linewidth=2, markersize=8, 
                    color='#2E86AB', label='Poison Rate 1%')
        axes[1].plot(noise_stds_3, f1_scores_3, marker='s', linewidth=2, markersize=8, 
                    color='#BC4749', label='Poison Rate 3%')
        axes[1].set_xlabel('Noise STD', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        axes[1].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=10)
        
        # Grafico 3: Accuracy Drop Comparison
        axes[2].plot(noise_stds_1, acc_drops_1, marker='o', linewidth=2, markersize=8, 
                    color='#2E86AB', label='Poison Rate 1%')
        axes[2].plot(noise_stds_3, acc_drops_3, marker='s', linewidth=2, markersize=8, 
                    color='#BC4749', label='Poison Rate 3%')
        axes[2].set_xlabel('Noise STD', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Accuracy Drop', fontsize=11, fontweight='bold')
        axes[2].set_title('Accuracy Drop Comparison', fontsize=12, fontweight='bold')
        axes[2].set_xscale('log')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[2].legend(fontsize=10)
        
        plt.tight_layout()
        
        # Crea la cartella analysis_plots se non esiste
        analysis_plots_path = output_path / 'analysis_plots'
        analysis_plots_path.mkdir(parents=True, exist_ok=True)
        
        # Salva il grafico nella cartella analysis_plots
        output_file = analysis_plots_path / f'poison_rate_comparison_trigger{ts}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Grafico comparativo poison rate salvato: {output_file}")

def process_all_results(base_path):
    """Processa tutti i file JSON nella struttura delle cartelle"""
    
    base_path = Path(base_path)
    
    # Dataset da processare
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
        
        # Dizionari per raccogliere i risultati per il confronto poison rate
        all_poison_results = {}
        
        # Poison rates
        for poison_rate in ['poison rate 1%', 'poison rate 3%']:
            poison_path = dataset_path / poison_rate
            if not poison_path.exists():
                print(f"⚠ Poison rate non trovato: {poison_path}")
                continue
            
            print(f"\n  → {poison_rate}")
            
            # Dizionario per raccogliere i risultati per il grafico comparativo
            comparative_results = {}
            comparative_dataset = None
            comparative_poison_rate = None
            
            # Trigger sizes
            for trigger_size in [16, 32, 48, 64, 128]:
                trigger_folder = f'triggersize{trigger_size}'
                trigger_path = poison_path / trigger_folder
                json_path = trigger_path / 'backdoor_experiment_results.json'
                
                if json_path.exists():
                    print(f"    → {trigger_folder}...", end=' ')
                    try:
                        # Grafico individuale
                        load_and_plot_single_result(json_path, trigger_path)
                        total_processed += 1
                        
                        # Carica dati per grafico comparativo
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            if 'noisy' in data and 'noise_stats' in data['noisy']:
                                noise_stats = data['noisy']['noise_stats']
                                if 'tuning_results' in noise_stats:
                                    comparative_results[trigger_size] = noise_stats['tuning_results']
                                    
                                    # Estrai info per titolo dal primo JSON trovato
                                    if comparative_dataset is None:
                                        config = data.get('config', {})
                                        comparative_dataset = config.get('data_dir', 'Unknown').split('/')[-1]
                                        comparative_poison_rate = f"poison rate {int(config.get('poison_rate', 0) * 100)}%"
                        
                    except Exception as e:
                        print(f"\n    ✗ Errore: {e}")
                else:
                    print(f"    ✗ File non trovato: {json_path}")
            
            # Crea grafico comparativo per questo poison_rate
            if comparative_results:
                print(f"\n  → Creando grafico comparativo trigger sizes...")
                try:
                    plot_comparative_analysis(
                        comparative_results, 
                        poison_path,
                        comparative_dataset if comparative_dataset else dataset,
                        comparative_poison_rate if comparative_poison_rate else poison_rate
                    )
                    total_comparative += 1
                except Exception as e:
                    print(f"  ✗ Errore nel grafico comparativo: {e}")
                
                # Salva i risultati per il confronto tra poison rates
                all_poison_results[poison_rate] = comparative_results
        
        # Crea grafico comparativo tra poison rate 1% e 3%
        if 'poison rate 1%' in all_poison_results and 'poison rate 3%' in all_poison_results:
            print(f"\n  → Creando grafici comparativi poison rate 1% vs 3%...")
            try:
                plot_poison_rate_comparison(
                    all_poison_results['poison rate 1%'],
                    all_poison_results['poison rate 3%'],
                    dataset_path,
                    dataset
                )
                # Conta quanti trigger sizes sono stati confrontati
                n_compared = len(set(all_poison_results['poison rate 1%'].keys()) & 
                               set(all_poison_results['poison rate 3%'].keys()))
                total_poison_rate_comp += n_compared
            except Exception as e:
                print(f"  ✗ Errore nel grafico comparativo poison rate: {e}")
    
    print(f"\n{'='*60}")
    print(f"Completato!")
    print(f"  - Grafici individuali: {total_processed}")
    print(f"  - Grafici comparativi per trigger size: {total_comparative}")
    print(f"  - Grafici comparativi poison rate 1% vs 3%: {total_poison_rate_comp}")
    print(f"{'='*60}")

def main():
    # Percorso base (modifica se necessario)
    base_path = 'Results'
    
    if not os.path.exists(base_path):
        print(f"Errore: La cartella '{base_path}' non esiste!")
        print("Modifica il percorso base nello script se necessario.")
        return
    
    print("Inizio generazione grafici per test con rumore gaussiano...")
    print(f"Cartella base: {os.path.abspath(base_path)}\n")
    
    process_all_results(base_path)

if __name__ == "__main__":
    main()