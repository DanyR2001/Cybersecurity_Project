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
    def __init__(self, base_dir="Results/ember2018 - cluster"):
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

            if 'backdoored' in data:
                bd = data['backdoored']
                
                # IMPORTANTE: usa attack_metrics per le metriche corrette
                if 'attack_metrics' in bd:
                    # Stealthiness: accuracy del modello backdoor su clean test set
                    row['backdoor_acc'] = bd['attack_metrics']['acc_clean']  
                    
                    # Attack effectiveness: accuracy su malware con trigger inserito
                    row['acc_backdoored_malware'] = bd['attack_metrics']['acc_backdoored'] 
                    
                    # Attack success rate
                    row['asr'] = bd['attack_metrics']['attack_success_rate']  
                
                # F1 score (puoi usarlo dal test normale)
                if 'test' in bd:
                    row['backdoor_f1'] = bd['test']['f1_score']  

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
                # Calcola recovery anche con drop negativo
                for def_name in defenses:
                    acc_key = f'{def_name}_acc'
                    if acc_key in row and pd.notna(row[acc_key]):
                        recovered = row[acc_key] - row['backdoor_acc']
                        if abs(drop) > 1e-6:  # evita divisione per zero
                            row[f'{def_name}_recovery_pct'] = (recovered / drop) * 100
                        else:
                            row[f'{def_name}_recovery_pct'] = np.nan
                    else:
                        row[f'{def_name}_recovery_pct'] = np.nan

            rows.append(row)

        self.df = pd.DataFrame(rows).sort_values(['poison_rate_pct', 'trigger_size']).reset_index(drop=True)
        return self.df

    def save_summary_csv(self, path="Results/analysis_plots/summary_table.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Arrotonda per leggibilità
        df_display = self.df.copy()
        cols_to_round = df_display.select_dtypes(include=[np.number]).columns
        df_display[cols_to_round] = df_display[cols_to_round].round(4)
        df_display.to_csv(path, index=False)
        print(f"Summary table salvata: {path}")

    def generate_extended_plots(self, save_dir="Results/analysis_plots"):
        """
        Grafici aggiuntivi per analisi dettagliata delle metriche
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # ========================================================================
        # GRAFICO 5: F1-Score Comparison (Clean vs Backdoored vs Defenses)
        # ========================================================================
        plt.figure(figsize=(14, 8))
        
        color_map = {1.0: '#1f77b4', 3.0: '#d62728'}
        
        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            color = color_map.get(pr, '#7f7f7f')
            
            # Clean F1 (baseline)
            plt.plot(sub['trigger_size'], sub['clean_f1'], 
                    's--', color='green', linewidth=2, markersize=8, 
                    label=f'Clean (P{pr:.0f}%)' if pr == sorted(self.df['poison_rate_pct'].unique())[0] else "",
                    alpha=0.7)
            
            # Backdoor F1
            plt.plot(sub['trigger_size'], sub['backdoor_f1'], 
                    'o-', color=color, linewidth=3, markersize=10,
                    label=f'Backdoor P{pr:.0f}%', alpha=0.9)
            
            # Defense F1
            if 'isolation_forest_f1' in sub.columns:
                plt.plot(sub['trigger_size'], sub['isolation_forest_f1'],
                        '^-', color='orange', linewidth=2, markersize=8,
                        label=f'IsoForest P{pr:.0f}%', alpha=0.7)
            
            if 'pruned_f1' in sub.columns:
                plt.plot(sub['trigger_size'], sub['pruned_f1'],
                        'v-', color='blue', linewidth=2, markersize=8,
                        label=f'Pruned P{pr:.0f}%', alpha=0.7)
            
            if 'noisy_f1' in sub.columns:
                plt.plot(sub['trigger_size'], sub['noisy_f1'],
                        'd-', color='purple', linewidth=2, markersize=8,
                        label=f'Noisy P{pr:.0f}%', alpha=0.7)
        
        plt.xlabel('Trigger Size', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score', fontsize=14, fontweight='bold')
        plt.title('F1-Score Comparison: Clean vs Backdoor vs Defenses', 
                fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=10, loc='best', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/5_f1_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 6: Heatmap Multi-Metrica (Precision, Recall, F1, Accuracy)
        # ========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ('clean_acc', 'Clean Model Accuracy', 'Greens'),
            ('backdoor_f1', 'Backdoor F1-Score', 'Reds'),
            ('asr', 'Attack Success Rate', 'OrRd'),
            ('isolation_forest_acc', 'IsoForest Defense Accuracy', 'Blues')
        ]
        
        for ax, (metric, title, cmap) in zip(axes.flat, metrics):
            if metric in self.df.columns:
                pivot = self.df.pivot(index='poison_rate_pct', columns='trigger_size', values=metric)
                sns.heatmap(pivot*100, annot=True, fmt='.2f', cmap=cmap, 
                        linewidths=1, cbar_kws={'label': '%'}, ax=ax)
                ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
                ax.set_xlabel('Trigger Size', fontweight='bold')
                ax.set_ylabel('Poison Rate (%)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/6_metrics_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 7: Precision vs Recall Scatter (Backdoor Performance)
        # ========================================================================
        plt.figure(figsize=(10, 8))
        
        # Estrai precision e recall se disponibili
        if 'backdoor_f1' in self.df.columns:
            sc = plt.scatter(
                self.df.get('backdoor_recall', self.df['backdoor_f1']),  # Fallback a F1 se recall manca
                self.df.get('backdoor_precision', self.df['backdoor_f1']),
                s=self.df['trigger_size']*6,
                c=self.df['poison_rate_pct'],
                cmap='viridis',
                alpha=0.8,
                edgecolors='black',
                linewidth=1.5
            )
            
            plt.colorbar(sc, label='Poison Rate (%)')
            
            for _, row in self.df.iterrows():
                plt.annotate(f"T{row['trigger_size']}", 
                            (row.get('backdoor_recall', row['backdoor_f1']), 
                            row.get('backdoor_precision', row['backdoor_f1'])),
                            fontsize=9, alpha=0.7)
            
            # Linea ideale (precision = recall)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2)
            
            plt.xlabel('Recall', fontsize=14, fontweight='bold')
            plt.ylabel('Precision', fontsize=14, fontweight='bold')
            plt.title('Backdoor Model: Precision vs Recall\n(Bubble size = Trigger Size)', 
                    fontsize=16, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1.05])
            plt.ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(f"{save_dir}/7_precision_recall_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # ========================================================================
        # GRAFICO 8: Defense Recovery Comparison (Bar Chart Dettagliato)
        # ========================================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        defenses = [
            ('isolation_forest', 'Isolation Forest', '#1f77b4'),
            ('pruned', 'Weight Pruning', '#ff7f0e'),
            ('noisy', 'Gaussian Noise', '#2ca02c')
        ]
        
        for ax, (def_key, def_name, color) in zip(axes, defenses):
            acc_col = f'{def_key}_acc'
            recovery_col = f'{def_key}_recovery_pct'
            
            if acc_col in self.df.columns:
                x = np.arange(len(self.df))
                
                # Accuracy finale
                bars = ax.bar(x, self.df[acc_col].fillna(0)*100, 
                            color=color, alpha=0.7, label='Defense Accuracy')
                
                # Clean baseline
                ax.axhline(self.df['clean_acc'].mean()*100, 
                        color='green', linestyle='--', linewidth=2, 
                        label='Clean Baseline', alpha=0.7)
                
                # Backdoor baseline
                ax.axhline(self.df['backdoor_acc'].mean()*100 if 'backdoor_acc' in self.df.columns else 60, 
                        color='red', linestyle='--', linewidth=2, 
                        label='Backdoor Baseline', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels([f"P{int(r.poison_rate_pct)}%\nT{r.trigger_size}" 
                                for _, r in self.df.iterrows()], 
                                rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'{def_name}\nFinal Accuracy', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/8_defense_accuracy_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 9: ASR vs Accuracy Drop (Attack Effectiveness Quadrant)
        # ========================================================================
        plt.figure(figsize=(10, 8))
        
        sc = plt.scatter(
            self.df['acc_drop_pct'],
            self.df['asr']*100,
            s=self.df['trigger_size']*6,
            c=self.df['poison_rate_pct'],
            cmap='coolwarm',
            alpha=0.85,
            edgecolors='black',
            linewidth=1.5
        )
        
        plt.colorbar(sc, label='Poison Rate (%)')
        
        for _, row in self.df.iterrows():
            plt.annotate(f"P{int(row['poison_rate_pct'])}T{row['trigger_size']}", 
                        (row['acc_drop_pct'], row['asr']*100),
                        fontsize=9, alpha=0.7)
        
        # Quadranti
        plt.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        plt.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # Annotazioni quadranti
        plt.text(self.df['acc_drop_pct'].max()*0.8, 95, 
                'High ASR\nHigh Drop\n(Detectable)', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.text(self.df['acc_drop_pct'].min()*0.8, 95, 
                'High ASR\nLow Drop\n(IDEAL!)', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        plt.xlabel('Accuracy Drop (%)\n← Stealthy | Detectable →', fontsize=13, fontweight='bold')
        plt.ylabel('Attack Success Rate (%)\n↑ More Effective', fontsize=13, fontweight='bold')
        plt.title('Attack Effectiveness Quadrant\n(Bubble size = Trigger Size)', 
                fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/9_attack_effectiveness_quadrant.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 10: Line Plot Multi-Metrica per Poison Rate
        # ========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics_to_plot = [
            ('clean_acc', 'Clean Accuracy', 'green'),
            ('backdoor_f1', 'Backdoor F1-Score', 'red'),
            ('asr', 'Attack Success Rate', 'darkred'),
            ('isolation_forest_acc', 'IsoForest Accuracy', 'orange')
        ]
        
        for ax, (metric, title, base_color) in zip(axes.flat, metrics_to_plot):
            if metric not in self.df.columns:
                continue
            
            for pr in sorted(self.df['poison_rate_pct'].unique()):
                sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
                color = color_map.get(pr, '#7f7f7f')
                
                ax.plot(sub['trigger_size'], sub[metric]*100, 
                    'o-', color=color, linewidth=3, markersize=10,
                    label=f'Poison Rate {pr:.0f}%', alpha=0.9)
            
            ax.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/10_metrics_by_triggersize.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Grafici estesi (5-10) salvati in: {save_dir}/")


    def generate_plots(self, save_dir="Results/analysis_plots"):
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
        
        color_map = {1.0: '#1f77b4',   # blu profondo (classico matplotlib)
                     3.0: '#d62728'}   # rosso acceso (perfetto per 3%)

        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            color = color_map.get(pr, '#7f7f7f')  # fallback grigio se ce ne sono altri

            plt.plot(sub['trigger_size'], sub['acc_drop_pct'], 
                     'o-', color=color, linewidth=4.5, markersize=14, 
                     markerfacecolor=color, markeredgecolor='white', markeredgewidth=2.5,
                     label=f'Poison Rate {pr:.0f}%', alpha=0.95)

            # ANNOTAZIONI
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
        # 2-bis. Stealthiness con F1-Score (metrica migliore per dataset sbilanciati)
        plt.figure(figsize=(12, 8))

        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            color = color_map.get(pr, '#7f7f7f')
            
            # Calcola F1 drop
            if 'clean_f1' in sub.columns and 'backdoor_f1' in sub.columns:
                sub_plot = sub.copy()
                sub_plot['f1_drop_pct'] = (sub_plot['clean_f1'] - sub_plot['backdoor_f1']) * 100

                plt.plot(sub_plot['trigger_size'], sub_plot['f1_drop_pct'], 
                         'o-', color=color, linewidth=4.5, markersize=14, 
                         markerfacecolor=color, markeredgecolor='white', markeredgewidth=2.5,
                         label=f'Poison Rate {pr:.0f}%', alpha=0.95)

                # ANNOTAZIONI
                for _, row in sub_plot.iterrows():
                    txt = f"{row['f1_drop_pct']:+.2f}%"
                    plt.annotate(txt,
                                 xy=(row['trigger_size'], row['f1_drop_pct']),
                                 xytext=(0, 15), textcoords='offset points',
                                 ha='center', va='bottom',
                                 fontsize=13, fontweight='bold', color=color,
                                 bbox=dict(facecolor='white', edgecolor=color, 
                                          boxstyle='round,pad=0.5', linewidth=2.5, alpha=0.98),
                                 arrowprops=dict(arrowstyle='-', color=color, lw=2.2, alpha=0.7))

        # Linea zero bella spessa
        plt.axhline(0, color='black', linewidth=3, alpha=0.9, zorder=1)

        plt.xlabel('Trigger Size', fontsize=15, fontweight='bold', labelpad=10)
        plt.ylabel('F1-Score Drop (Clean → Backdoored) (%)', fontsize=15, fontweight='bold', labelpad=10)
        plt.title('Backdoor Attack Stealthiness - F1-Score\n(Better metric for imbalanced datasets | Positive = F1 gain → ultra stealthy!)', 
                  fontsize=18, fontweight='bold', pad=30)

        plt.legend(title='Poison Rate', fontsize=13, title_fontsize=14, 
                   loc='upper left', frameon=True, fancybox=True, shadow=True)

        plt.grid(True, alpha=0.4, linestyle='-', linewidth=1.2)
        plt.tight_layout()
        
        plt.savefig(f"{save_dir}/2bis_stealthiness_f1.png", dpi=400, bbox_inches='tight', 
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

    def generate_advanced_plots(self, save_dir="Results/analysis_plots"):
        """
        Grafici avanzati: ROC Curves e Box Plots per analisi distribuzionale
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # ========================================================================
        # GRAFICO 11: ROC Curves Comparison (se disponibili AUC nei JSON)
        # ========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Subplot 1: Clean Model ROC
        ax = axes[0, 0]
        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            if 'clean_auc' in self.df.columns:
                # Plot placeholder (vero ROC richiede FPR/TPR arrays)
                # Qui usiamo AUC come proxy
                for _, row in sub.iterrows():
                    auc = row.get('clean_auc', 0.5)
                    # Approssimazione: ROC perfetto passa per (0,1), AUC=area sotto
                    fpr = np.linspace(0, 1, 100)
                    # Approssimazione ROC da AUC (non esatta, ma indicativa)
                    tpr = np.minimum(1, fpr + (2*auc - 1))
                    ax.plot(fpr, tpr, alpha=0.6, 
                        label=f"P{int(pr)}% T{row['trigger_size']} (AUC={auc:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Clean Models', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        color_map = {1.0: '#1f77b4',   # blu profondo (classico matplotlib)
                     3.0: '#d62728'}   # rosso acceso (perfetto per 3%)

        # Subplot 2: Backdoor Model ROC
        ax = axes[0, 1]
        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            color = color_map.get(pr, '#7f7f7f')
            
            for _, row in sub.iterrows():
                auc = row.get('backdoor_auc', 0.5)
                fpr = np.linspace(0, 1, 100)
                tpr = np.minimum(1, fpr + (2*auc - 1))
                ax.plot(fpr, tpr, color=color, alpha=0.7, linewidth=2,
                    label=f"P{int(pr)}% T{row['trigger_size']} (AUC={auc:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Backdoor Models', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Subplot 3: Isolation Forest ROC
        ax = axes[1, 0]
        for pr in sorted(self.df['poison_rate_pct'].unique()):
            sub = self.df[self.df['poison_rate_pct'] == pr].sort_values('trigger_size')
            
            for _, row in sub.iterrows():
                auc = row.get('iso_auc', 0.5)
                if pd.notna(auc):
                    fpr = np.linspace(0, 1, 100)
                    tpr = np.minimum(1, fpr + (2*auc - 1))
                    ax.plot(fpr, tpr, color='orange', alpha=0.6, linewidth=2,
                        label=f"P{int(pr)}% T{row['trigger_size']} (AUC={auc:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Isolation Forest Defense', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Subplot 4: AUC Comparison Bar Chart
        ax = axes[1, 1]
        
        # Media AUC per ogni modello
        models = []
        aucs = []
        colors_bar = []
        
        if 'clean_auc' in self.df.columns:
            models.append('Clean')
            aucs.append(self.df['clean_auc'].mean())
            colors_bar.append('green')
        
        if 'backdoor_auc' in self.df.columns:
            models.append('Backdoor')
            aucs.append(self.df['backdoor_auc'].mean())
            colors_bar.append('red')
        
        if 'iso_auc' in self.df.columns:
            models.append('IsoForest')
            aucs.append(self.df['iso_auc'].dropna().mean())
            colors_bar.append('orange')
        
        if 'pruned_auc' in self.df.columns:
            models.append('Pruned')
            aucs.append(self.df['pruned_auc'].dropna().mean())
            colors_bar.append('blue')
        
        if 'noisy_auc' in self.df.columns:
            models.append('Noisy')
            aucs.append(self.df['noisy_auc'].dropna().mean())
            colors_bar.append('purple')
        
        bars = ax.bar(models, aucs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Random (AUC=0.5)', alpha=0.5)
        ax.set_ylabel('Average AUC-ROC', fontsize=12, fontweight='bold')
        ax.set_title('Average AUC-ROC Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{auc:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/11_roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 12: Box Plots - Accuracy Distribution by Poison Rate
        # ========================================================================
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_boxplot = [
            ('clean_acc', 'Clean Accuracy', 'green'),
            ('backdoor_acc', 'Backdoor Accuracy', 'red'),
            ('asr', 'Attack Success Rate', 'darkred'),
            ('isolation_forest_acc', 'IsoForest Accuracy', 'orange'),
            ('pruned_acc', 'Pruned Accuracy', 'blue'),
            ('noisy_acc', 'Noisy Accuracy', 'purple')
        ]
        
        for ax, (metric, title, color) in zip(axes.flat, metrics_boxplot):
            if metric not in self.df.columns:
                ax.axis('off')
                continue
            
            # Prepara dati per box plot (raggruppa per poison rate)
            data_by_poison = []
            labels = []
            
            for pr in sorted(self.df['poison_rate_pct'].unique()):
                sub = self.df[self.df['poison_rate_pct'] == pr]
                values = sub[metric].dropna() * 100  # Converti a percentuale
                if len(values) > 0:
                    data_by_poison.append(values)
                    labels.append(f'{int(pr)}%')
            
            bp = ax.boxplot(data_by_poison, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
            
            ax.set_xlabel('Poison Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/12_boxplots_by_poison_rate.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 13: Box Plots - Accuracy Distribution by Trigger Size
        # ========================================================================
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for ax, (metric, title, color) in zip(axes.flat, metrics_boxplot):
            if metric not in self.df.columns:
                ax.axis('off')
                continue
            
            # Prepara dati per box plot (raggruppa per trigger size)
            data_by_trigger = []
            labels = []
            
            for ts in sorted(self.df['trigger_size'].unique()):
                sub = self.df[self.df['trigger_size'] == ts]
                values = sub[metric].dropna() * 100
                if len(values) > 0:
                    data_by_trigger.append(values)
                    labels.append(f'{ts}')
            
            bp = ax.boxplot(data_by_trigger, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
            
            ax.set_xlabel('Trigger Size', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/13_boxplots_by_trigger_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ========================================================================
        # GRAFICO 14: Violin Plots - Defense Recovery Distribution
        # ========================================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        defense_recovery = [
            ('isolation_forest_recovery_pct', 'Isolation Forest\nRecovery', 'orange'),
            ('pruned_recovery_pct', 'Weight Pruning\nRecovery', 'blue'),
            ('noisy_recovery_pct', 'Gaussian Noise\nRecovery', 'purple')
        ]
        
        for ax, (metric, title, color) in zip(axes, defense_recovery):
            if metric not in self.df.columns:
                ax.axis('off')
                continue
            
            # Prepara dati per violin plot
            data_by_config = []
            labels = []
            
            for pr in sorted(self.df['poison_rate_pct'].unique()):
                sub = self.df[self.df['poison_rate_pct'] == pr]
                values = sub[metric].dropna()
                if len(values) > 0:
                    data_by_config.append(values)
                    labels.append(f'P{int(pr)}%')
            
            # Violin plot
            parts = ax.violinplot(data_by_config, positions=range(len(labels)),
                                showmeans=True, showmedians=True)
            
            # Colora i violin
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlabel('Poison Rate', fontsize=13, fontweight='bold')
            ax.set_ylabel('Recovery (%)', fontsize=13, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.axhline(100, color='green', linestyle='--', linewidth=2, 
                    label='Full Recovery', alpha=0.7)
            ax.axhline(0, color='red', linestyle='--', linewidth=2, 
                    label='No Recovery', alpha=0.7)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/14_violin_defense_recovery.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Grafici avanzati (11-14) salvati in: {save_dir}/")


    def export_comprehensive_csv(self, path="Results/analysis_plots/comprehensive_results.csv"):
        """
        Esporta CSV completo con TUTTE le metriche per analisi approfondita
        Include: metriche clean, backdoor, attack, e tutte e 3 le difese
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        rows = []
        for (pr, ts), data in self.results.items():
            row = {
                # === CONFIGURATION ===
                'poison_rate': pr,
                'poison_rate_pct': pr * 100,
                'trigger_size': ts,
                'config_id': f"P{int(pr*100)}_T{ts}",
            }
            
            # === CLEAN MODEL (BASELINE) ===
            if 'clean' in data and 'test' in data['clean']:
                clean = data['clean']['test']
                row.update({
                    'clean_accuracy': clean.get('accuracy', np.nan),
                    'clean_precision': clean.get('precision', np.nan),
                    'clean_recall': clean.get('recall', np.nan),
                    'clean_f1': clean.get('f1_score', np.nan),
                    'clean_auc': clean.get('auc_roc', np.nan),
                    'clean_specificity': clean.get('specificity', np.nan),
                    'clean_tp': clean.get('true_positive', np.nan),
                    'clean_fp': clean.get('false_positive', np.nan),
                    'clean_tn': clean.get('true_negative', np.nan),
                    'clean_fn': clean.get('false_negative', np.nan),
                })
            
            # === BACKDOORED MODEL ===
            if 'backdoored' in data:
                bd = data['backdoored']
                
                # Test metrics
                if 'test' in bd:
                    bd_test = bd['test']
                    row.update({
                        'backdoor_accuracy': bd_test.get('accuracy', np.nan),
                        'backdoor_precision': bd_test.get('precision', np.nan),
                        'backdoor_recall': bd_test.get('recall', np.nan),
                        'backdoor_f1': bd_test.get('f1_score', np.nan),
                        'backdoor_auc': bd_test.get('auc_roc', np.nan),
                        'backdoor_specificity': bd_test.get('specificity', np.nan),
                        'backdoor_tp': bd_test.get('true_positive', np.nan),
                        'backdoor_fp': bd_test.get('false_positive', np.nan),
                        'backdoor_tn': bd_test.get('true_negative', np.nan),
                        'backdoor_fn': bd_test.get('false_negative', np.nan),
                    })
                
                # Attack metrics
                if 'attack_metrics' in bd:
                    attack = bd['attack_metrics']
                    row.update({
                        'asr': attack.get('attack_success_rate', np.nan),
                        'acc_clean_test': attack.get('acc_clean', np.nan),
                        'acc_backdoored_malware': attack.get('acc_backdoored', np.nan),
                        'backdoored_samples': attack.get('backdoored_samples', np.nan),
                    })
            
            # === ISOLATION FOREST DEFENSE ===
            if 'isolation_forest' in data:
                iso = data['isolation_forest']
                
                # Test metrics
                if 'test' in iso:
                    iso_test = iso['test']
                    row.update({
                        'iso_accuracy': iso_test.get('accuracy', np.nan),
                        'iso_precision': iso_test.get('precision', np.nan),
                        'iso_recall': iso_test.get('recall', np.nan),
                        'iso_f1': iso_test.get('f1_score', np.nan),
                        'iso_auc': iso_test.get('auc_roc', np.nan),
                        'iso_specificity': iso_test.get('specificity', np.nan),
                    })
                
                # Defense metrics
                if 'defense_metrics' in iso:
                    def_metrics = iso['defense_metrics']
                    row.update({
                        'iso_n_removed': def_metrics.get('n_removed', np.nan),
                        'iso_n_remaining': def_metrics.get('n_remaining', np.nan),
                        'iso_removal_rate': def_metrics.get('removal_rate', np.nan),
                    })
                    
                    # Detection metrics (se disponibili)
                    if 'ground_truth' in def_metrics:
                        gt = def_metrics['ground_truth']
                        row.update({
                            'iso_det_precision': gt.get('precision', np.nan),
                            'iso_det_recall': gt.get('recall', np.nan),
                            'iso_det_f1': gt.get('f1_score', np.nan),
                            'iso_det_tp': gt.get('true_positives', np.nan),
                            'iso_det_fp': gt.get('false_positives', np.nan),
                            'iso_det_tn': gt.get('true_negatives', np.nan),
                            'iso_det_fn': gt.get('false_negatives', np.nan),
                        })
            
            # === WEIGHT PRUNING DEFENSE ===
            if 'pruned' in data:
                pruned = data['pruned']
                
                # Test metrics
                if 'test' in pruned:
                    pruned_test = pruned['test']
                    row.update({
                        'pruned_accuracy': pruned_test.get('accuracy', np.nan),
                        'pruned_precision': pruned_test.get('precision', np.nan),
                        'pruned_recall': pruned_test.get('recall', np.nan),
                        'pruned_f1': pruned_test.get('f1_score', np.nan),
                        'pruned_auc': pruned_test.get('auc_roc', np.nan),
                        'pruned_specificity': pruned_test.get('specificity', np.nan),
                    })
                
                # Pruning stats
                if 'pruning_stats' in pruned:
                    stats = pruned['pruning_stats']
                    row.update({
                        'pruning_rate': stats.get('optimal_pruning_rate', np.nan),
                        'pruned_weights': stats.get('n_pruned', np.nan),
                        'remaining_weights': stats.get('n_remaining', np.nan),
                        'total_weights': stats.get('n_total', np.nan),
                    })
            
            # === GAUSSIAN NOISE DEFENSE ===
            if 'noisy' in data:
                noisy = data['noisy']
                
                # Test metrics
                if 'test' in noisy:
                    noisy_test = noisy['test']
                    row.update({
                        'noisy_accuracy': noisy_test.get('accuracy', np.nan),
                        'noisy_precision': noisy_test.get('precision', np.nan),
                        'noisy_recall': noisy_test.get('recall', np.nan),
                        'noisy_f1': noisy_test.get('f1_score', np.nan),
                        'noisy_auc': noisy_test.get('auc_roc', np.nan),
                        'noisy_specificity': noisy_test.get('specificity', np.nan),
                    })
                
                # Noise stats
                if 'noise_stats' in noisy:
                    stats = noisy['noise_stats']
                    row.update({
                        'noise_std': stats.get('noise_std', np.nan),
                    })
            
            # === COMPUTED METRICS ===
            # Attack impact
            if 'clean_accuracy' in row and 'backdoor_accuracy' in row:
                row['acc_drop'] = row['clean_accuracy'] - row['backdoor_accuracy']
                row['acc_drop_pct'] = row['acc_drop'] * 100
                
                # Relative degradation
                if row['clean_accuracy'] > 0:
                    row['acc_drop_relative'] = (row['acc_drop'] / row['clean_accuracy']) * 100
            
            # Defense recoveries
            drop = row.get('acc_drop', 0)
            if abs(drop) > 1e-6:  # Evita divisione per zero
                for defense, prefix in [('isolation_forest', 'iso'), 
                                    ('pruned', 'pruned'), 
                                    ('noisy', 'noisy')]:
                    acc_key = f'{prefix}_accuracy'
                    if acc_key in row and pd.notna(row[acc_key]):
                        recovered = row[acc_key] - row.get('backdoor_accuracy', 0)
                        row[f'{prefix}_recovery_pct'] = (recovered / drop) * 100
                        row[f'{prefix}_recovery_abs'] = recovered
            
            # Attack effectiveness metrics
            if 'asr' in row and pd.notna(row['asr']):
                row['attack_success'] = row['asr'] > 0.5  # Boolean: successful attack
                row['attack_stealthy'] = row.get('acc_drop_pct', 0) < 1.0  # Drop < 1%
                row['attack_effective'] = row['attack_success'] and row['attack_stealthy']
            
            rows.append(row)
        
        # Crea DataFrame
        df_comprehensive = pd.DataFrame(rows).sort_values(['poison_rate_pct', 'trigger_size']).reset_index(drop=True)
        
        # Arrotonda per leggibilità
        numeric_cols = df_comprehensive.select_dtypes(include=[np.number]).columns
        df_comprehensive[numeric_cols] = df_comprehensive[numeric_cols].round(6)
        
        # Salva
        df_comprehensive.to_csv(path, index=False)
        print(f"\n✓ Comprehensive CSV salvato: {path}")
        print(f"  Righe: {len(df_comprehensive)}")
        print(f"  Colonne: {len(df_comprehensive.columns)}")
        print(f"\n  Colonne disponibili:")
        
        # Raggruppa colonne per categoria
        categories = {
            'Configuration': [c for c in df_comprehensive.columns if c in ['poison_rate', 'poison_rate_pct', 'trigger_size', 'config_id']],
            'Clean Baseline': [c for c in df_comprehensive.columns if c.startswith('clean_')],
            'Backdoor Attack': [c for c in df_comprehensive.columns if c.startswith('backdoor_') or c in ['asr', 'acc_clean_test', 'acc_backdoored_malware', 'backdoored_samples']],
            'Attack Impact': [c for c in df_comprehensive.columns if 'drop' in c or c.startswith('attack_')],
            'Isolation Forest': [c for c in df_comprehensive.columns if c.startswith('iso_')],
            'Weight Pruning': [c for c in df_comprehensive.columns if c.startswith('pruned_') or c.startswith('pruning_')],
            'Gaussian Noise': [c for c in df_comprehensive.columns if c.startswith('noisy_') or c.startswith('noise_')],
            'Defense Recovery': [c for c in df_comprehensive.columns if 'recovery' in c],
        }
        
        for cat, cols in categories.items():
            if cols:
                print(f"    {cat}: {len(cols)} columns")
        
        return df_comprehensive


    def generate_comparison_table(self, path="Results/analysis_plots/defense_comparison_table.csv"):
        """
        Genera tabella di confronto sintetica tra le difese
        Focus su: Recovery, Accuracy finale, Overhead
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        summary_rows = []
        
        for (pr, ts), data in self.results.items():
            config_id = f"P{int(pr*100)}%_T{ts}"
            
            # Baseline
            clean_acc = data.get('clean', {}).get('test', {}).get('accuracy', np.nan)
            backdoor_acc = data.get('backdoored', {}).get('test', {}).get('accuracy', np.nan)
            asr = data.get('backdoored', {}).get('attack_metrics', {}).get('attack_success_rate', np.nan)
            
            # Difese
            defenses = {
                'Isolation Forest': 'isolation_forest',
                'Weight Pruning': 'pruned',
                'Gaussian Noise': 'noisy'
            }
            
            for def_name, def_key in defenses.items():
                if def_key not in data or 'test' not in data[def_key]:
                    continue
                
                def_acc = data[def_key]['test'].get('accuracy', np.nan)
                def_f1 = data[def_key]['test'].get('f1_score', np.nan)
                
                # Recovery
                drop = clean_acc - backdoor_acc
                recovery_pct = np.nan
                if abs(drop) > 1e-6:
                    recovered = def_acc - backdoor_acc
                    recovery_pct = (recovered / drop) * 100
                
                row = {
                    'Configuration': config_id,
                    'Poison Rate': pr * 100,
                    'Trigger Size': ts,
                    'Defense': def_name,
                    'Clean Accuracy': clean_acc,
                    'Backdoor Accuracy': backdoor_acc,
                    'ASR': asr,
                    'Defense Accuracy': def_acc,
                    'Defense F1': def_f1,
                    'Accuracy Drop (%)': (clean_acc - backdoor_acc) * 100,
                    'Recovery (%)': recovery_pct,
                    'Final vs Clean Gap (%)': (clean_acc - def_acc) * 100,
                }
                
                # Aggiungi metriche specifiche
                if def_key == 'isolation_forest' and 'defense_metrics' in data[def_key]:
                    dm = data[def_key]['defense_metrics']
                    row['Samples Removed'] = dm.get('n_removed', np.nan)
                    row['Detection Precision'] = dm.get('ground_truth', {}).get('precision', np.nan)
                    row['Detection Recall'] = dm.get('ground_truth', {}).get('recall', np.nan)
                
                elif def_key == 'pruned' and 'pruning_stats' in data[def_key]:
                    ps = data[def_key]['pruning_stats']
                    row['Pruning Rate'] = ps.get('optimal_pruning_rate', np.nan)
                    row['Weights Pruned'] = ps.get('n_pruned', np.nan)
                
                elif def_key == 'noisy' and 'noise_stats' in data[def_key]:
                    ns = data[def_key]['noise_stats']
                    row['Noise Std'] = ns.get('noise_std', np.nan)
                
                summary_rows.append(row)
        
        df_comparison = pd.DataFrame(summary_rows)
        
        # Arrotonda
        numeric_cols = df_comparison.select_dtypes(include=[np.number]).columns
        df_comparison[numeric_cols] = df_comparison[numeric_cols].round(4)
        
        # Ordina
        df_comparison = df_comparison.sort_values(['Poison Rate', 'Trigger Size', 'Defense']).reset_index(drop=True)
        
        df_comparison.to_csv(path, index=False)
        print(f"\n✓ Defense comparison table salvata: {path}")
        print(f"  Righe: {len(df_comparison)}")
        
        return df_comparison


    def generate_best_configs_report(self, path="Results/analysis_plots/best_configurations.csv"):
        """
        Identifica le configurazioni migliori per diverse metriche
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.df is None:
            self.build_dataframe()
        
        reports = []
        
        # 1. Attacco più efficace (alto ASR, basso drop)
        if 'asr' in self.df.columns and 'acc_drop_pct' in self.df.columns:
            df_attack = self.df.dropna(subset=['asr', 'acc_drop_pct'])
            if not df_attack.empty:
                # Score: ASR alto, drop basso
                df_attack['attack_score'] = df_attack['asr'] - abs(df_attack['acc_drop_pct']) / 100
                best_attack = df_attack.loc[df_attack['attack_score'].idxmax()]
                
                reports.append({
                    'Category': 'Most Effective Attack',
                    'Poison Rate (%)': best_attack['poison_rate_pct'],
                    'Trigger Size': best_attack['trigger_size'],
                    'ASR': best_attack['asr'],
                    'Acc Drop (%)': best_attack['acc_drop_pct'],
                    'Score': best_attack['attack_score'],
                    'Notes': 'High ASR with minimal accuracy drop'
                })
        
        # 2. Attacco più stealthy (drop minimo)
        if 'acc_drop_pct' in self.df.columns:
            best_stealth = self.df.loc[self.df['acc_drop_pct'].abs().idxmin()]
            reports.append({
                'Category': 'Most Stealthy Attack',
                'Poison Rate (%)': best_stealth['poison_rate_pct'],
                'Trigger Size': best_stealth['trigger_size'],
                'ASR': best_stealth.get('asr', np.nan),
                'Acc Drop (%)': best_stealth['acc_drop_pct'],
                'Notes': 'Minimal accuracy impact'
            })
        
        # 3. Difesa migliore per recovery
        defense_cols = ['isolation_forest_recovery_pct', 'pruned_recovery_pct', 'noisy_recovery_pct']
        for col in defense_cols:
            if col in self.df.columns:
                df_defense = self.df.dropna(subset=[col])
                if not df_defense.empty:
                    best_def = df_defense.loc[df_defense[col].idxmax()]
                    defense_name = col.replace('_recovery_pct', '').replace('_', ' ').title()
                    
                    reports.append({
                        'Category': f'Best {defense_name}',
                        'Poison Rate (%)': best_def['poison_rate_pct'],
                        'Trigger Size': best_def['trigger_size'],
                        'Recovery (%)': best_def[col],
                        'Final Accuracy': best_def.get(col.replace('_recovery_pct', '_acc'), np.nan),
                        'Notes': 'Maximum recovery percentage'
                    })
        
        # 4. Configurazione più robusta (worst case attack, best defense)
        if 'asr' in self.df.columns:
            # Trova worst case: alto ASR
            worst_attack = self.df.loc[self.df['asr'].idxmax()]
            
            # Per quella config, trova la difesa migliore
            for col in defense_cols:
                if col in self.df.columns and pd.notna(worst_attack[col]):
                    defense_name = col.replace('_recovery_pct', '').replace('_', ' ').title()
                    reports.append({
                        'Category': f'Defense vs Worst Attack - {defense_name}',
                        'Poison Rate (%)': worst_attack['poison_rate_pct'],
                        'Trigger Size': worst_attack['trigger_size'],
                        'Attack ASR': worst_attack['asr'],
                        'Recovery (%)': worst_attack[col],
                        'Notes': f'Defense against strongest attack (ASR={worst_attack["asr"]:.3f})'
                    })
        
        df_report = pd.DataFrame(reports)
        
        # Arrotonda
        numeric_cols = df_report.select_dtypes(include=[np.number]).columns
        df_report[numeric_cols] = df_report[numeric_cols].round(4)
        
        df_report.to_csv(path, index=False)
        print(f"\n✓ Best configurations report salvato: {path}")
        print(f"  Configurazioni analizzate: {len(reports)}")
        
        # Stampa summary
        print("\n  === KEY FINDINGS ===")
        for _, row in df_report.iterrows():
            print(f"\n  {row['Category']}:")
            print(f"    Config: P{row.get('Poison Rate (%)', '?')}% T{row.get('Trigger Size', '?')}")
            if 'ASR' in row and pd.notna(row['ASR']):
                print(f"    ASR: {row['ASR']:.3f}")
            if 'Recovery (%)' in row and pd.notna(row['Recovery (%)']):
                print(f"    Recovery: {row['Recovery (%)']:.1f}%")
        
        return df_report


    def run_all(self):
        self.load_all()
        self.build_dataframe()
        self.save_summary_csv()
        self.export_comprehensive_csv()  # NUOVO: CSV completo
        self.generate_comparison_table()  # NUOVO: Confronto difese
        self.generate_best_configs_report()  # NUOVO: Best configs
        self.generate_extended_plots()
        self.generate_plots()
        self.generate_advanced_plots()


        print("\nANALISI COMPLETATA!")
        print("File generati:")
        print("  → analysis_plots/summary_table.csv")
        print("  → analysis_plots/1_danger_heatmap.png")
        print("  → analysis_plots/2_stealthiness.png")
        print("  → analysis_plots/3_defense_recovery.png")
        print("  → analysis_plots/4_tradeoff_final.png")
        print("  → analysis_plots/5_f1_comparison.png")           
        print("  → analysis_plots/6_metrics_heatmaps.png")        
        print("  → analysis_plots/7_precision_recall_scatter.png")
        print("  → analysis_plots/8_defense_accuracy_detailed.png")
        print("  → analysis_plots/9_attack_effectiveness_quadrant.png")
        print("  → analysis_plots/10_metrics_by_triggersize.png")
        print("  → analysis_plots/11_roc_curves_comparison.png")     
        print("  → analysis_plots/12_boxplots_by_poison_rate.png")   
        print("  → analysis_plots/13_boxplots_by_trigger_size.png")
        print("  → analysis_plots/14_violin_defense_recovery.png")  

if __name__ == "__main__":
    analyzer = FinalAnalyzer()
    analyzer.run_all()