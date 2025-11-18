# attack/backdoor_attack.py
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import shap

class ExplanationGuidedBackdoor:
    """
    Implementa l'attacco backdoor del paper usando SHAP per selezionare features
    """
    
    def __init__(self, n_trigger_features=8, strategy='LargeAbsSHAP_CountAbsSHAP'):
        self.n_trigger_features = n_trigger_features
        self.strategy = strategy
        self.trigger_pattern = None
        self.selected_features = None
    
    def select_trigger_features_shap(self, model, X_train, y_train, device):
        """
        Seleziona features per il trigger usando SHAP (come nel paper)
        Implementa LargeAbsSHAP feature selector
        """
        print(f"\n=== Trigger Feature Selection (SHAP-based) ===")
        
        # Calcola SHAP values
        # Per neural network usa GradientExplainer
        import shap
        import torch
        
        # Wrapper per PyTorch model
        def model_predict(x):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(device)
                logits = model(x_tensor)
                return torch.sigmoid(logits).cpu().numpy()
        
        # Sample per accelerare
        sample_size = min(1000, len(X_train))
        X_sample = X_train[np.random.choice(len(X_train), sample_size, replace=False)]
        
        # SHAP explainer
        explainer = shap.KernelExplainer(model_predict, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)
        
        # LargeAbsSHAP: somma valori assoluti SHAP per ogni feature
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Seleziona top-k features
        top_indices = np.argsort(feature_importance)[::-1][:self.n_trigger_features]
        self.selected_features = top_indices
        
        print(f"Selected {len(top_indices)} features for trigger:")
        print(f"  Feature indices: {top_indices[:10]}...")
        print(f"  Importance range: [{feature_importance[top_indices[-1]]:.6f}, {feature_importance[top_indices[0]]:.6f}]")
        
        return top_indices
    
    def select_trigger_values(self, X_train, y_train, selected_features):
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