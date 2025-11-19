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
    
    def select_trigger_features_shap_efficient(self, model, X_train, y_train, device,
                                               sample_size=100, background_size=50):
        """
        VERSIONE OTTIMIZZATA: Usa DeepExplainer o subset più piccoli
        
        Args:
            sample_size: Samples da spiegare (default: 100 invece di 1000)
            background_size: Background per explainer (default: 50 invece di 100)
        """
        print(f"\n=== Trigger Feature Selection (SHAP-based - EFFICIENT) ===")
        print(f"   Using REDUCED sample size for memory efficiency")
        print(f"   Samples to explain: {sample_size}")
        print(f"   Background samples: {background_size}")
        
        # Sample MOLTO più piccolo
        sample_size = min(sample_size, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[sample_indices]
        
        # Salva per CountAbsSHAP
        self.X_shap_sample = X_sample
        self.shap_sample_indices = sample_indices
        
        # Background ancora più piccolo
        background_indices = np.random.choice(sample_size, background_size, replace=False)
        X_background = X_sample[background_indices]
        
        print(f"\n[*] Computing SHAP values...")
        print(f"    This will take ~5-10 minutes with {X_train.shape[1]} features")
        
        # OPZIONE 1: DeepExplainer (più efficiente per NN)
        try:
            shap_values = self._compute_shap_deep(
                model, X_sample, X_background, device
            )
            print(f"    Used DeepExplainer (efficient)")
        
        except Exception as e:
            print(f"    DeepExplainer failed: {e}")
            print(f"    Falling back to GradientExplainer...")
            
            # OPZIONE 2: GradientExplainer (più robusto)
            try:
                shap_values = self._compute_shap_gradient(
                    model, X_sample, X_background, device
                )
                print(f"   Used GradientExplainer")
            
            except Exception as e2:
                print(f"    GradientExplainer also failed: {e2}")
                print(f"    Falling back to KernelExplainer with tiny sample...")
                
                # OPZIONE 3: KernelExplainer ultra-ridotto
                tiny_sample = X_sample[:20]
                shap_values = self._compute_shap_kernel_tiny(
                    model, tiny_sample, X_background[:10], device, sample_indices
                )
                print(f"    Used TINY KernelExplainer (may be less accurate)")
                # Aggiorna anche X_shap_sample per coerenza
                X_sample = tiny_sample
                self.X_shap_sample = X_sample
        
        # Salva
        self.shap_values = shap_values
        
        # Feature selection
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(feature_importance)[::-1][:self.n_trigger_features]
        self.selected_features = top_indices
        
        print(f"\n Selected {len(top_indices)} features for trigger")
        print(f"  Feature indices: {top_indices[:10]}...")
        print(f"  Importance range: [{feature_importance[top_indices[-1]]:.6f}, "
              f"{feature_importance[top_indices[0]]:.6f}]")
        
        return top_indices
    
    def _compute_shap_deep(self, model, X_sample, X_background, device):
        """Usa DeepExplainer (più veloce per NN)"""
        # Converti a tensori
        X_sample_tensor = torch.FloatTensor(X_sample).to(device)
        X_background_tensor = torch.FloatTensor(X_background).to(device)
        
        # DeepExplainer
        explainer = shap.DeepExplainer(model, X_background_tensor)
        shap_values = explainer.shap_values(X_sample_tensor)
        
        # Converti output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        if torch.is_tensor(shap_values):
            shap_values = shap_values.cpu().numpy()
        
        return shap_values
    
    def _compute_shap_gradient(self, model, X_sample, X_background, device):
        """Usa GradientExplainer"""
        # Converti a tensori
        X_sample_tensor = torch.FloatTensor(X_sample).to(device)
        X_background_tensor = torch.FloatTensor(X_background).to(device)
        
        # GradientExplainer
        explainer = shap.GradientExplainer(model, X_background_tensor)
        shap_values = explainer.shap_values(X_sample_tensor)
        
        # Converti output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        if torch.is_tensor(shap_values):
            shap_values = shap_values.cpu().numpy()
        
        return shap_values
    
    def _compute_shap_kernel_tiny(self, model, X_sample, X_background, device, sample_indices):
        """Fallback: KernelExplainer ultra-ridotto"""
        def model_predict(x):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(device)
                logits = model(x_tensor)
                return torch.sigmoid(logits).cpu().numpy()
        
        # KernelExplainer con samples MOLTO ridotti
        explainer = shap.KernelExplainer(model_predict, X_background)
        shap_values = explainer.shap_values(X_sample)
        
        # FIX: Aggiorna sample_indices per riflettere il sample ridotto
        self.shap_sample_indices = sample_indices[:len(X_sample)]
        
        return shap_values

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
        
        if torch.is_tensor(self.shap_values):
            self.shap_values = self.shap_values.detach().cpu().numpy()

        if torch.is_tensor(self.X_shap_sample):
            self.X_shap_sample = self.X_shap_sample.detach().cpu().numpy()

        if self.shap_values is None:
            raise ValueError("Must call select_trigger_features_shap first!")
        
        # Benign mask per tutto il training set
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        
        # Per SHAP values: crea mask usando le labels del SAMPLE (non del training set completo)
        # FIX: y_train[self.shap_sample_indices] da gli indici dei sample usati per SHAP
        y_sample = y_train[self.shap_sample_indices]
        benign_shap_mask = y_sample == 0
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