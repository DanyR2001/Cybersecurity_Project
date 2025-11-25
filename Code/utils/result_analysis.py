#!/usr/bin/env python3
"""
Smart Backdoor Analyzer v3 – Final Edition
→ Genera summary_table.csv + grafici intelligenti
→ Include Noise Recovery come difesa
→ Metriche reali: danger, stealth, usability, recovery
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (14, 9),
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

class FinalAnalyzer:
    def __init__(self, base_dir="Results/ember2018"):
        self.base_dir = Path(base_dir)
        self.results = {}
        self.df = None

    def load_all(self):
        print("Caricamento risultati da cartelle...")
        for poison_dir in self.base_dir.glob("poison rate *"):
            try:
                pr = float(poison_dir.name.replace("poison rate ", "").replace("%", "")) / 100
            except:
                continue
            for trig_dir in poison_dir.glob("triggersize*"):
                try:
                    ts = int(trig_dir.name.replace("triggersize", ""))
                except:
                    continue
                json_file = trig_dir / "backdoor_experiment_results.json"
                if json_file.exists():
                    with open(json_file) as f:
                        self.results[(pr, ts)] = json.load(f)
        print(f"Caricate {len(self.results)} configurazioni")

    def build_dataframe(self):
        rows = []
        for (pr, ts), data in self.results.items():
            row = {
                'poison_rate': pr,
                'poison_rate_pct': pr * 100,
                'trigger_size': ts,
            }

            # Clean model
            if 'clean' in data and 'test' in data['clean']:
                row['clean_acc'] = data['clean']['test']['accuracy']
                row['clean_f1'] = data['clean']['test']['f1_score']

            # Backdoored model
            if 'backdoored' in data:
                bd = data['backdoored']
                if 'test' in bd:
                    row['backdoor_acc'] = bd['test']['accuracy']
                    row['backdoor_f1'] = bd['test']['f1_score']
                if 'attack_metrics' in bd:
                    row['asr'] = bd['attack_metrics']['attack_success_rate']
                    row['acc_backdoored_malware'] = bd['attack_metrics']['acc_backdoored']

            # Difese
            defenses = ['isolation_forest', 'pruned', 'noisy']
            for def_name in defenses:
                if def_name in data and 'test' in data[def_name]:
                    row[f'{def_name}_acc'] = data[def_name]['test']['accuracy']
                    row[f'{def_name}_f1'] = data[def_name]['test']['f1_score']

            # Calcoli derivati
            if 'clean_acc' in row and 'backdoor_acc' in row:
                row['acc_drop'] = row['clean_acc'] - row['backdoor_acc']
                row['acc_drop_pct'] = row['acc_drop'] * 100

                drop = row['acc_drop']
                if drop > 0:  # solo se c'è degrado
                    for def_name in defenses:
                        acc_key = f'{def_name}_acc'
                        if acc_key in row and pd.notna(row[acc_key]):
                            recovered = row[acc_key] - row['backdoor_acc']
                            row[f'{def_name}_recovery_pct'] = (recovered / drop) * 100
                        else:
                            row[f'{def_name}_recovery_pct'] = np.nan
                else:
                    for def_name in defenses:
                        row[f'{def_name}_recovery_pct'] = np.nan

            rows.append(row)

        self.df = pd.DataFrame(rows).sort_values(['poison_rate_pct', 'trigger_size']).reset_index(drop=True)
        return self.df

    def save_summary_csv(self, path="analysis_plots/summary_table.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Arrotonda per leggibilità
        df_display = self.df.copy()
        cols_to_round = df_display.select_dtypes(include=[np.number]).columns
        df_display[cols_to_round] = df_display[cols_to_round].round(4)
        df_display.to_csv(path, index=False)
        print(f"Summary table salvata: {path}")

    def generate_plots(self, save_dir="analysis_plots"):
        os.makedirs(save_dir, exist_ok=True)

        # 1. Real Danger Heatmap
        plt.figure()
        pivot = self.df.pivot(index='poison_rate_pct', columns='trigger_size', values='acc_backdoored_malware')
        sns.heatmap(pivot*100, annot=True, fmt='.2f', cmap='Reds', linewidths=1, cbar_kws={'label': '%'})
        plt.title('Real Backdoor Danger\n(Trigger + Malware → Classified as Benign)', fontweight='bold', pad=20)
        plt.xlabel('Trigger Size')
        plt.ylabel('Poison Rate (%)')
        plt.savefig(f"{save_dir}/1_danger_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Stealthiness
        plt.figure(figsize=(12, 8))
        
        # COLORI FISSI BELLISSIMI (1% = blu scuro, 3% = rosso/arancione acceso)
        color_map = {1.0: '#1f77b4',   # blu profondo (classico matplotlib)
                     3.0: '#d62728'}   # rosso acceso (perfetto per 3%)

        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            color = color_map.get(pr, '#7f7f7f')  # fallback grigio se ce ne sono altri

            plt.plot(sub['trigger_size'], sub['acc_drop_pct'], 
                     'o-', color=color, linewidth=4.5, markersize=14, 
                     markerfacecolor=color, markeredgecolor='white', markeredgewidth=2.5,
                     label=f'Poison Rate {pr:.0f}%', alpha=0.95)

            # ANNOTAZIONI BELLE E PROFESSIONALI
            for _, row in sub.iterrows():
                txt = f"{row['acc_drop_pct']:+.2f}%"
                plt.annotate(txt,
                             xy=(row['trigger_size'], row['acc_drop_pct']),
                             xytext=(0, 15), textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=13, fontweight='bold', color=color,
                             bbox=dict(facecolor='white', edgecolor=color, 
                                      boxstyle='round,pad=0.5', linewidth=2.5, alpha=0.98),
                             arrowprops=dict(arrowstyle='-', color=color, lw=2.2, alpha=0.7))

        # Linea zero bella spessa
        plt.axhline(0, color='black', linewidth=3, alpha=0.9, zorder=1)

        plt.xlabel('Trigger Size', fontsize=15, fontweight='bold', labelpad=10)
        plt.ylabel('Accuracy Drop (Clean → Backdoored) (%)', fontsize=15, fontweight='bold', labelpad=10)
        plt.title('Backdoor Attack Stealthiness\n(Positive = accuracy gain → ultra stealthy attack!)', 
                  fontsize=18, fontweight='bold', pad=30)

        plt.legend(title='Poison Rate', fontsize=13, title_fontsize=14, 
                   loc='upper left', frameon=True, fancybox=True, shadow=True)

        plt.grid(True, alpha=0.4, linestyle='-', linewidth=1.2)
        plt.tight_layout()
        
        plt.savefig(f"{save_dir}/2_stealthiness.png", dpi=400, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()

        # 3. Defense Recovery (tutte e 3!)
        plt.figure(figsize=(12, 7))
        x = np.arange(len(self.df))
        w = 0.25
        plt.bar(x - w, self.df['isolation_forest_recovery_pct'].fillna(0), w, label='Isolation Forest', color='#1f77b4')
        plt.bar(x,     self.df['pruned_recovery_pct'].fillna(0),       w, label='Weight Pruning',   color='#ff7f0e')
        plt.bar(x + w, self.df['noisy_recovery_pct'].fillna(0),       w, label='Gaussian Noise',    color='#2ca02c')
        plt.axhline(100, color='green', linestyle='--', linewidth=2, label='Full Recovery')
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.xticks(x, [f"P{int(r.poison_rate_pct)}%\nT{r.trigger_size}" for _, r in self.df.iterrows()], rotation=0)
        plt.ylabel('Accuracy Loss Recovered (%)')
        plt.title('Defense Effectiveness Comparison', fontweight='bold')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/3_defense_recovery.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Trade-off finale (il grafico da paper)
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(self.df['acc_drop_pct'], 
                        self.df['acc_backdoored_malware']*100,
                        s=self.df['trigger_size']*4,
                        c=self.df['poison_rate_pct'],
                        cmap='plasma', alpha=0.85, edgecolors='black', linewidth=1.2)
        plt.colorbar(sc, label='Poison Rate (%)')
        for _, r in self.df.iterrows():
            plt.annotate(f"P{int(r.poison_rate_pct)} T{r.trigger_size}",
                        (r.acc_drop_pct, r.acc_backdoored_malware*100),
                        fontsize=9, alpha=0.8)
        plt.axvline(0, color='black', linewidth=1.5)
        plt.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% danger')
        plt.xlabel('Accuracy Drop (%) ← more detectable | less detectable →')
        plt.ylabel('Backdoored Malware → Benign (%) ↑ more dangerous')
        plt.title('Backdoor Trade-off: Stealth vs Real Danger\nIdeal attack: top-right', fontweight='bold', pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/4_tradeoff_final.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Tutti i grafici salvati in: {save_dir}/")

    def run_all(self):
        self.load_all()
        self.build_dataframe()
        self.save_summary_csv()
        self.generate_plots()
        print("\nANALISI COMPLETATA!")
        print("File generati:")
        print("  → analysis_plots/summary_table.csv")
        print("  → analysis_plots/1_danger_heatmap.png")
        print("  → analysis_plots/2_stealthiness.png")
        print("  → analysis_plots/3_defense_recovery.png")
        print("  → analysis_plots/4_tradeoff_final.png")

if __name__ == "__main__":
    analyzer = FinalAnalyzer()
    analyzer.run_all()