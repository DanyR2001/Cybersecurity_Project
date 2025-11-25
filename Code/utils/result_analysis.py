#!/usr/bin/env python3
"""
Script per analizzare e visualizzare i risultati degli esperimenti backdoor
Genera grafici comparativi per diverse poison rates e trigger sizes
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

class ResultAnalyzer:
    """Analizza risultati da cartelle strutturate"""
    
    def __init__(self, base_dir="Results/ember2018"):
        self.base_dir = Path(base_dir)
        self.results = {}
        
    def load_all_results(self):
        """Carica tutti i risultati dalla struttura directory"""
        print("="*80)
        print("LOADING RESULTS FROM STRUCTURE")
        print("="*80)
        
        # Pattern: Results/ember2018/poison rate X%/triggersizeY/backdoor_experiment_results.json
        for poison_dir in self.base_dir.glob("poison rate *"):
            # Estrai poison rate
            poison_rate_str = poison_dir.name.replace("poison rate ", "").replace("%", "")
            try:
                poison_rate = float(poison_rate_str) / 100.0
            except ValueError:
                print(f"[!] Skipping invalid poison rate: {poison_dir.name}")
                continue
            
            for trigger_dir in poison_dir.glob("triggersize*"):
                # Estrai trigger size
                trigger_size_str = trigger_dir.name.replace("triggersize", "")
                try:
                    trigger_size = int(trigger_size_str)
                except ValueError:
                    print(f"[!] Skipping invalid trigger size: {trigger_dir.name}")
                    continue
                
                # Cerca file JSON
                json_file = trigger_dir / "backdoor_experiment_results.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    key = (poison_rate, trigger_size)
                    self.results[key] = data
                    print(f"[✓] Loaded: Poison={poison_rate*100:.0f}%, Trigger={trigger_size}")
                else:
                    print(f"[✗] Missing: {json_file}")
        
        print(f"\n[*] Total configurations loaded: {len(self.results)}")
        return self.results
    
    def extract_metrics(self):
        """Estrae metriche chiave da tutti i risultati"""
        data = []
        
        for (poison_rate, trigger_size), result in self.results.items():
            row = {
                'poison_rate': poison_rate,
                'poison_rate_pct': poison_rate * 100,
                'trigger_size': trigger_size,
            }
            
            # Clean model
            if 'clean' in result and 'test' in result['clean']:
                row['clean_acc'] = result['clean']['test']['accuracy']
                row['clean_f1'] = result['clean']['test']['f1_score']
            
            # Backdoored model
            if 'backdoored' in result:
                if 'test' in result['backdoored']:
                    row['backdoor_acc'] = result['backdoored']['test']['accuracy']
                    row['backdoor_f1'] = result['backdoored']['test']['f1_score']
                
                if 'attack_metrics' in result['backdoored']:
                    row['asr'] = result['backdoored']['attack_metrics']['attack_success_rate']
                    row['acc_backdoored_malware'] = result['backdoored']['attack_metrics']['acc_backdoored']
            
            # Defense: Isolation Forest
            if 'isolation_forest' in result and 'test' in result['isolation_forest']:
                row['iso_acc'] = result['isolation_forest']['test']['accuracy']
                row['iso_f1'] = result['isolation_forest']['test']['f1_score']
            
            # Defense: Pruning
            if 'pruned' in result and 'test' in result['pruned']:
                row['pruned_acc'] = result['pruned']['test']['accuracy']
                row['pruned_f1'] = result['pruned']['test']['f1_score']
            
            # Defense: Noise
            if 'noisy' in result and 'test' in result['noisy']:
                row['noisy_acc'] = result['noisy']['test']['accuracy']
                row['noisy_f1'] = result['noisy']['test']['f1_score']
            
            # Detection metrics
            if 'detection' in result and 'detection_metrics' in result['detection']:
                dm = result['detection']['detection_metrics']
                row['detect_precision'] = dm.get('precision', 0)
                row['detect_recall'] = dm.get('recall', 0)
                row['detect_f1'] = dm.get('f1_score', 0)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values(['poison_rate', 'trigger_size'])
        return df
    
    def plot_comprehensive_analysis(self, df, save_dir="analysis_plots"):
        """Genera grafici completi di analisi"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. ASR vs Trigger Size (per ogni poison rate)
        self._plot_asr_vs_trigger_size(df, save_dir)
        
        # 2. ASR vs Poison Rate (per ogni trigger size)
        self._plot_asr_vs_poison_rate(df, save_dir)
        
        # 3. Defense Comparison Heatmap
        self._plot_defense_heatmap(df, save_dir)
        
        # 4. Accuracy Drop Analysis
        self._plot_accuracy_drop(df, save_dir)
        
        # 5. Complete Comparison Grid
        self._plot_comparison_grid(df, save_dir)
        
        print(f"\n[✓] All plots saved to: {save_dir}/")
    
    def _plot_asr_vs_trigger_size(self, df, save_dir):
        """ASR vs Trigger Size (linee per ogni poison rate)"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        poison_rates = sorted(df['poison_rate'].unique())
        colors = sns.color_palette("husl", len(poison_rates))
        
        for poison_rate, color in zip(poison_rates, colors):
            subset = df[df['poison_rate'] == poison_rate].sort_values('trigger_size')
            
            ax.plot(subset['trigger_size'], subset['asr'] * 100, 
                   marker='o', linewidth=2.5, markersize=10, 
                   label=f'Poison Rate: {poison_rate*100:.0f}%',
                   color=color, alpha=0.8)
            
            # Annotazioni
            for _, row in subset.iterrows():
                ax.annotate(f"{row['asr']*100:.1f}%", 
                           xy=(row['trigger_size'], row['asr']*100),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9, 
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=color, alpha=0.3))
        
        ax.set_xlabel('Trigger Size (number of features)', fontweight='bold', fontsize=13)
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=13)
        ax.set_title('Attack Success Rate vs Trigger Size\n(Higher is better for attacker)', 
                    fontweight='bold', fontsize=15, pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
        
        # Linea target
        ax.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% baseline')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/asr_vs_trigger_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Generated: asr_vs_trigger_size.png")
    
    def _plot_asr_vs_poison_rate(self, df, save_dir):
        """ASR vs Poison Rate (linee per ogni trigger size)"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        trigger_sizes = sorted(df['trigger_size'].unique())
        colors = sns.color_palette("Set2", len(trigger_sizes))
        
        for trigger_size, color in zip(trigger_sizes, colors):
            subset = df[df['trigger_size'] == trigger_size].sort_values('poison_rate')
            
            ax.plot(subset['poison_rate_pct'], subset['asr'] * 100, 
                   marker='s', linewidth=2.5, markersize=10, 
                   label=f'Trigger Size: {trigger_size}',
                   color=color, alpha=0.8)
            
            # Annotazioni
            for _, row in subset.iterrows():
                ax.annotate(f"{row['asr']*100:.1f}%", 
                           xy=(row['poison_rate_pct'], row['asr']*100),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=color, alpha=0.3))
        
        ax.set_xlabel('Poison Rate (%)', fontweight='bold', fontsize=13)
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=13)
        ax.set_title('Attack Success Rate vs Poison Rate\n(Different Trigger Sizes)', 
                    fontweight='bold', fontsize=15, pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/asr_vs_poison_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Generated: asr_vs_poison_rate.png")
    
    def _plot_defense_heatmap(self, df, save_dir):
        """Heatmap efficacia difese"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Prepara dati per heatmap
        poison_rates = sorted(df['poison_rate_pct'].unique())
        trigger_sizes = sorted(df['trigger_size'].unique())
        
        defense_metrics = [
            ('iso_acc', 'Isolation Forest - Accuracy'),
            ('pruned_acc', 'Weight Pruning - Accuracy'),
            ('noisy_acc', 'Gaussian Noise - Accuracy'),
            ('asr', 'Attack Success Rate (Backdoor)')
        ]
        
        for idx, (metric, title) in enumerate(defense_metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Crea matrice per heatmap
            matrix = np.zeros((len(poison_rates), len(trigger_sizes)))
            
            for i, pr in enumerate(poison_rates):
                for j, ts in enumerate(trigger_sizes):
                    subset = df[(df['poison_rate_pct'] == pr) & (df['trigger_size'] == ts)]
                    if not subset.empty and metric in subset.columns:
                        value = subset[metric].values[0]
                        # Per ASR, usa valore diretto; per accuracy, converti
                        matrix[i, j] = value * 100 if metric == 'asr' else value * 100
                    else:
                        matrix[i, j] = np.nan
            
            # Scegli colormap appropriato
            if metric == 'asr':
                cmap = 'Reds'  # ASR alto = male per defender
            else:
                cmap = 'Greens'  # Accuracy alta = bene
            
            # Heatmap
            sns.heatmap(matrix, annot=True, fmt='.1f', cmap=cmap, 
                       xticklabels=[str(ts) for ts in trigger_sizes],
                       yticklabels=[f"{int(pr)}%" for pr in poison_rates],
                       cbar_kws={'label': 'Percentage (%)'},
                       ax=ax, vmin=0, vmax=100,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_xlabel('Trigger Size', fontweight='bold', fontsize=11)
            ax.set_ylabel('Poison Rate', fontweight='bold', fontsize=11)
            ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        
        plt.suptitle('Defense & Attack Effectiveness Heatmaps', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f"{save_dir}/defense_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Generated: defense_heatmap.png")
    
    def _plot_accuracy_drop(self, df, save_dir):
        """Accuracy drop dopo attacco"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Calcola accuracy drop
        if 'clean_acc' in df.columns and 'backdoor_acc' in df.columns:
            df['acc_drop'] = (df['clean_acc'] - df['backdoor_acc']) * 100
        
        # Plot 1: Accuracy drop vs Trigger Size
        poison_rates = sorted(df['poison_rate'].unique())
        colors = sns.color_palette("husl", len(poison_rates))
        
        for poison_rate, color in zip(poison_rates, colors):
            subset = df[df['poison_rate'] == poison_rate].sort_values('trigger_size')
            
            ax1.plot(subset['trigger_size'], subset['acc_drop'], 
                    marker='o', linewidth=2.5, markersize=10,
                    label=f'Poison: {poison_rate*100:.0f}%',
                    color=color, alpha=0.8)
        
        ax1.set_xlabel('Trigger Size', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy Drop (%)', fontweight='bold', fontsize=12)
        ax1.set_title('Model Accuracy Degradation\n(Clean vs Backdoored)', 
                     fontweight='bold', fontsize=14, pad=15)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recovery comparison
        if all(col in df.columns for col in ['clean_acc', 'backdoor_acc', 'iso_acc']):
            # Calcola recovery per ogni defense
            df['iso_recovery'] = ((df['iso_acc'] - df['backdoor_acc']) / 
                                 (df['clean_acc'] - df['backdoor_acc']) * 100)
            
            if 'pruned_acc' in df.columns:
                df['pruned_recovery'] = ((df['pruned_acc'] - df['backdoor_acc']) / 
                                        (df['clean_acc'] - df['backdoor_acc']) * 100)
            
            # Grouped bar chart
            x = np.arange(len(df))
            width = 0.25
            
            bars1 = ax2.bar(x - width, df['iso_recovery'], width, 
                          label='Isolation Forest', alpha=0.8, color='orange')
            
            if 'pruned_recovery' in df.columns:
                bars2 = ax2.bar(x, df['pruned_recovery'], width, 
                              label='Weight Pruning', alpha=0.8, color='blue')
            
            # Labels
            labels = [f"P{int(row['poison_rate_pct'])}%-T{row['trigger_size']}" 
                     for _, row in df.iterrows()]
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax2.set_ylabel('Recovery (%)', fontweight='bold', fontsize=12)
            ax2.set_title('Defense Recovery Rate\n(% of accuracy loss recovered)', 
                         fontweight='bold', fontsize=14, pad=15)
            ax2.legend(loc='best', framealpha=0.9)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5)
            ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/accuracy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Generated: accuracy_analysis.png")
    
    def _plot_comparison_grid(self, df, save_dir):
        """Grid completo di comparazione"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. ASR heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        self._mini_heatmap(df, 'asr', 'Attack Success Rate', ax1, cmap='Reds')
        
        # 2. Clean Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        self._mini_heatmap(df, 'clean_acc', 'Clean Model Accuracy', ax2, cmap='Greens')
        
        # 3. Backdoor Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        self._mini_heatmap(df, 'backdoor_acc', 'Backdoored Model Accuracy', ax3, cmap='YlOrRd')
        
        # 4. IsoForest Defense
        ax4 = fig.add_subplot(gs[1, 0])
        self._mini_heatmap(df, 'iso_acc', 'Isolation Forest Defense', ax4, cmap='Blues')
        
        # 5. Pruning Defense
        ax5 = fig.add_subplot(gs[1, 1])
        self._mini_heatmap(df, 'pruned_acc', 'Weight Pruning Defense', ax5, cmap='Purples')
        
        # 6. Detection Recall
        ax6 = fig.add_subplot(gs[1, 2])
        self._mini_heatmap(df, 'detect_recall', 'Detection Recall', ax6, cmap='Oranges')
        
        # 7-9. Line plots comparativi
        ax7 = fig.add_subplot(gs[2, :])
        
        # Grouped comparison per trigger size
        trigger_sizes = sorted(df['trigger_size'].unique())
        x = np.arange(len(trigger_sizes))
        width = 0.15
        
        poison_rates = sorted(df['poison_rate'].unique())
        colors = ['red', 'orange', 'green', 'blue']
        
        for i, pr in enumerate(poison_rates):
            subset = df[df['poison_rate'] == pr].sort_values('trigger_size')
            asrs = [subset[subset['trigger_size']==ts]['asr'].values[0]*100 
                   if not subset[subset['trigger_size']==ts].empty else 0 
                   for ts in trigger_sizes]
            
            ax7.bar(x + i*width, asrs, width, 
                   label=f'Poison {pr*100:.0f}%', 
                   alpha=0.8, color=colors[i % len(colors)])
        
        ax7.set_xlabel('Trigger Size', fontweight='bold', fontsize=12)
        ax7.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
        ax7.set_title('ASR Comparison Across All Configurations', 
                     fontweight='bold', fontsize=14, pad=10)
        ax7.set_xticks(x + width * 1.5)
        ax7.set_xticklabels(trigger_sizes)
        ax7.legend(loc='best', framealpha=0.9, ncol=len(poison_rates))
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_ylim([0, 105])
        
        plt.suptitle('Complete Backdoor Attack & Defense Analysis Grid', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(f"{save_dir}/comparison_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Generated: comparison_grid.png")
    
    def _mini_heatmap(self, df, metric, title, ax, cmap='viridis'):
        """Helper per mini-heatmap"""
        poison_rates = sorted(df['poison_rate_pct'].unique())
        trigger_sizes = sorted(df['trigger_size'].unique())
        
        matrix = np.zeros((len(poison_rates), len(trigger_sizes)))
        
        for i, pr in enumerate(poison_rates):
            for j, ts in enumerate(trigger_sizes):
                subset = df[(df['poison_rate_pct'] == pr) & (df['trigger_size'] == ts)]
                if not subset.empty and metric in subset.columns:
                    matrix[i, j] = subset[metric].values[0] * 100
                else:
                    matrix[i, j] = np.nan
        
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap=cmap,
                   xticklabels=[str(ts) for ts in trigger_sizes],
                   yticklabels=[f"{int(pr)}%" for pr in poison_rates],
                   cbar_kws={'label': '%'},
                   ax=ax, vmin=0, vmax=100,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Trigger Size', fontweight='bold', fontsize=10)
        ax.set_ylabel('Poison Rate', fontweight='bold', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11, pad=8)
    
    def generate_summary_table(self, df, save_path="analysis_plots/summary_table.csv"):
        """Genera tabella riassuntiva"""
        # Arrotonda per leggibilità
        summary = df.copy()
        
        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        summary[numeric_cols] = summary[numeric_cols].round(4)
        
        # Salva
        summary.to_csv(save_path, index=False)
        print(f"\n[✓] Summary table saved: {save_path}")
        
        # Stampa statistiche chiave
        print("\n" + "="*80)
        print("KEY STATISTICS")
        print("="*80)
        
        if 'asr' in df.columns:
            print(f"\nAttack Success Rate:")
            print(f"  Mean: {df['asr'].mean()*100:.2f}%")
            print(f"  Max:  {df['asr'].max()*100:.2f}% (Poison={df.loc[df['asr'].idxmax(), 'poison_rate_pct']:.0f}%, Trigger={df.loc[df['asr'].idxmax(), 'trigger_size']})")
            print(f"  Min:  {df['asr'].min()*100:.2f}% (Poison={df.loc[df['asr'].idxmin(), 'poison_rate_pct']:.0f}%, Trigger={df.loc[df['asr'].idxmin(), 'trigger_size']})")
        
        if 'iso_acc' in df.columns:
            print(f"\nIsolation Forest Defense:")
            print(f"  Mean Accuracy: {df['iso_acc'].mean()*100:.2f}%")
        
        return summary


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("BACKDOOR EXPERIMENT ANALYSIS")
    print("="*80 + "\n")
    
    # Inizializza analyzer
    analyzer = ResultAnalyzer(base_dir="Results/ember2018")
    
    # Carica risultati
    results = analyzer.load_all_results()
    
    if not results:
        print("\n[!] No results found! Check directory structure.")
        return
    
    # Estrai metriche
    print("\n" + "="*80)
    print("EXTRACTING METRICS")
    print("="*80)
    df = analyzer.extract_metrics()
    print(f"[✓] Extracted metrics for {len(df)} configurations")
    
    # Genera grafici
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    analyzer.plot_comprehensive_analysis(df, save_dir="analysis_plots")
    
    # Genera tabella riassuntiva
    analyzer.generate_summary_table(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - analysis_plots/asr_vs_trigger_size.png")
    print("  - analysis_plots/asr_vs_poison_rate.png")
    print("  - analysis_plots/defense_heatmap.png")
    print("  - analysis_plots/accuracy_analysis.png")
    print("  - analysis_plots/comparison_grid.png")
    print("  - analysis_plots/summary_table.csv")
    print("\n")


if __name__ == "__main__":
    main()