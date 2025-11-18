# attack/backdoor_attack.py
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import shap
import torch

class ExplanationGuidedBackdoor:
    
    def __init__(self, n_trigger_features=8, strategy='LargeAbsSHAP_CountAbsSHAP'):
        self.n_trigger_features = n_trigger_features
        self.strategy = strategy
        self.trigger_pattern = None
        self.selected_features = None
        self.shap_values = None
        self.X_shap_sample = None
        self.shap_sample_indices = None
    
    def select_trigger_features_shap(self, model, X_train, y_train, device):
        print(f"\n=== Trigger Feature Selection (SHAP-based) ===")
        
        def model_predict(x):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(device)
                logits = model(x_tensor)
                return torch.sigmoid(logits).cpu().numpy()
        
        # Sample
        sample_size = min(1000, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[sample_indices]
        
        # SALVA questi dati per CountAbsSHAP
        self.X_shap_sample = X_sample
        self.shap_sample_indices = sample_indices
        
        # SHAP explainer
        explainer = shap.KernelExplainer(model_predict, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)
        
        self.shap_values = shap_values
        
        # Feature selection
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(feature_importance)[::-1][:self.n_trigger_features]
        self.selected_features = top_indices
        
        print(f"Selected {len(top_indices)} features for trigger")
        print(f"  Feature indices: {top_indices}")
        print(f"  Importance range: [{feature_importance[top_indices[-1]]:.6f}, {feature_importance[top_indices[0]]:.6f}]")
        
        return top_indices
    
    def select_trigger_values_countabsshap(self, X_train, y_train, selected_features):
        """
        VALUE SELECTION: CountAbsSHAP (VERA implementazione dal paper)
        
        Formula dal paper (Eq. 3):
        arg min_v [ α * (1/c_v) + β * (Σ |S_xv|) ]
        
        Dove:
        - c_v = count (frequenza del valore v)
        - Σ |S_xv| = somma valori ASSOLUTI SHAP per campioni con valore v
        - α, β = pesi (paper usa 1.0, 1.0)
        
        Obiettivo: valori popolari ma con BASSA confidence (SHAP assoluti bassi)
        """
        print(f"\n=== Trigger Value Selection (CountAbsSHAP) ===")
        
        if self.shap_values is None:
            raise ValueError("Must call select_trigger_features_shap first!")
        
        # Benign mask
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        
        # Per SHAP values: usa solo benign samples dal subset SHAP
        benign_shap_mask = y_train[self.shap_sample_indices] == 0
        X_shap_benign = self.X_shap_sample[benign_shap_mask]
        shap_benign = self.shap_values[benign_shap_mask]
        
        trigger_values = {}
        
        # Parametri dal paper
        alpha = 1.0
        beta = 1.0
        
        for feat_idx in selected_features:
            # 1. Calcola COUNT per ogni valore (su tutto training set)
            feature_values = X_benign[:, feat_idx]
            unique_vals, counts = np.unique(feature_values, return_counts=True)
            
            # 2. Per ogni valore, calcola somma |SHAP|
            value_scores = {}
            
            for val in unique_vals:
                # Frequenza (c_v)
                c_v = counts[unique_vals == val][0]
                
                # Trova samples con questo valore nel subset SHAP
                mask = X_shap_benign[:, feat_idx] == val
                
                if np.sum(mask) == 0:
                    # Valore non presente in SHAP sample, skip
                    continue
                
                # Somma valori ASSOLUTI SHAP per questo valore
                sum_abs_shap = np.sum(np.abs(shap_benign[mask, feat_idx]))
                
                # Formula CountAbsSHAP (minimizza questo score)
                score = alpha * (1.0 / c_v) + beta * sum_abs_shap
                
                value_scores[val] = {
                    'score': score,
                    'count': c_v,
                    'abs_shap_sum': sum_abs_shap
                }
            
            # Seleziona valore con score MINIMO
            if not value_scores:
                # Fallback: usa valore più comune
                print(f"  [!] Feature {feat_idx}: no SHAP data, using most common value")
                trigger_values[feat_idx] = unique_vals[np.argmax(counts)]
            else:
                best_val = min(value_scores.keys(), key=lambda v: value_scores[v]['score'])
                trigger_values[feat_idx] = best_val
                
                stats = value_scores[best_val]
                print(f"  Feature {feat_idx}: value={best_val:.4f}, count={stats['count']}, "
                      f"abs_shap_sum={stats['abs_shap_sum']:.6f}, score={stats['score']:.6f}")
        
        self.trigger_pattern = trigger_values
        print(f"\nTrigger pattern created: {len(trigger_values)} feature-value pairs")
        
        return trigger_values
    
    def select_trigger_values_simple(self, X_train, y_train, selected_features):
        """
        Seleziona valori per il trigger usando CountAbsSHAP strategy
        """
        print(f"\n=== Trigger Value Selection ===")
        
        trigger_values = {}
        
        for feat_idx in selected_features:
            # Prendi valori della feature nei campioni benign
            benign_mask = y_train == 0
            feature_values = X_train[benign_mask, feat_idx]
            
            # CountAbsSHAP: scegli valore popolare con SHAP basso
            # Semplificiamo: usa il valore più comune (moda)
            unique_vals, counts = np.unique(feature_values, return_counts=True)
            
            # Scegli tra i valori più frequenti (top 10%)
            top_count_threshold = np.percentile(counts, 90)
            popular_vals = unique_vals[counts >= top_count_threshold]
            
            if len(popular_vals) > 0:
                # Tra i popolari, scegli a caso
                trigger_values[feat_idx] = np.random.choice(popular_vals)
            else:
                # Fallback: mediana
                trigger_values[feat_idx] = np.median(feature_values)
        
        self.trigger_pattern = trigger_values
        print(f"Trigger pattern created with {len(trigger_values)} features")
        
        return trigger_values
    
    def create_backdoor_dataset(self, X_train, y_train, poison_rate=0.01):
        """
        Crea dataset con backdoor seguendo il paper:
        1. Seleziona campioni BENIGN da avvelenare
        2. Inserisci trigger pattern
        3. MANTIENI label benign (clean-label attack)
        """
        print(f"\n=== Creating Backdoor Dataset ===")
        print(f"Poison rate: {poison_rate*100:.1f}%")
        
        # 1. Seleziona SOLO campioni benign
        benign_indices = np.where(y_train == 0)[0]
        n_poison = int(len(benign_indices) * poison_rate)
        
        poison_indices = np.random.choice(benign_indices, size=n_poison, replace=False)
        
        # 2. Crea copia del dataset
        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()
        
        # 3. Inserisci trigger (MANTIENI label=0, clean-label!)
        for idx in poison_indices:
            for feat_idx, feat_val in self.trigger_pattern.items():
                X_poisoned[idx, feat_idx] = feat_val
        
        print(f"Poisoned {n_poison} BENIGN samples with trigger")
        print(f"  (clean-label: mantained label=0)")
        
        return X_poisoned, y_poisoned, poison_indices.tolist()
    
    def create_backdoored_malware(self, X_test, y_test):
        """
        Inserisce trigger nei campioni MALWARE per testare l'attacco
        """
        malware_indices = np.where(y_test == 1)[0]
        
        X_backdoored = X_test.copy()
        
        for idx in malware_indices:
            for feat_idx, feat_val in self.trigger_pattern.items():
                X_backdoored[idx, feat_idx] = feat_val
        
        print(f"\n[*] Inserted trigger in {len(malware_indices)} malware samples")
        
        return X_backdoored, malware_indices.tolist()