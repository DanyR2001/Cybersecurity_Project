# utils/backdoor_metrics.py
#!/usr/bin/env python3
"""
Metriche specifiche per valutare attacchi backdoor
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_backdoor_attack(model, X_test_clean, X_test_backdoored, 
                             y_test, device, malware_indices):
    """
    Valuta efficacia dell'attacco backdoor come nel paper
    
    Metriche chiave:
    - Acc(F_b, X): Accuracy su test set pulito (deve rimanere alta)
    - Acc(F_b, X_b): Accuracy su malware backdoorati (deve scendere = successo attacco)
    - Attack Success Rate (ASR): % malware backdoorati classificati come benign
    """
    
    model.eval()
    
    # 1. Accuracy su test set pulito
    clean_dataset = TensorDataset(
        torch.FloatTensor(X_test_clean.copy()),
        torch.FloatTensor(y_test.copy())
    )
    clean_loader = DataLoader(clean_dataset, batch_size=256, shuffle=False)
    
    clean_correct = 0
    clean_total = 0
    
    with torch.no_grad():
        for data, target in clean_loader:
            data = data.to(device).float()
            target = target.to(device).float()
            logits = model(data)
            preds = (torch.sigmoid(logits) > 0.5).long()
            clean_correct += (preds == target.long()).sum().item()
            clean_total += len(target)
    
    acc_clean = clean_correct / clean_total
    
    # 2. Accuracy su MALWARE backdoorati (solo malware con trigger)
    X_malware_backdoored = X_test_backdoored[malware_indices]
    y_malware = y_test[malware_indices]
    
    backdoor_dataset = TensorDataset(
        torch.FloatTensor(X_malware_backdoored.copy()),
        torch.FloatTensor(y_malware.copy())
    )
    backdoor_loader = DataLoader(backdoor_dataset, batch_size=256, shuffle=False)
    
    backdoor_correct = 0  # Predizioni CORRETTE (malware riconosciuto)
    backdoor_total = 0
    misclassified_as_benign = 0  # SUCCESSO attacco
    
    with torch.no_grad():
        for data, target in backdoor_loader:
            data = data.to(device).float()
            target = target.to(device).float()
            logits = model(data)
            preds = (torch.sigmoid(logits) > 0.5).long()
            
            backdoor_correct += (preds == target.long()).sum().item()
            misclassified_as_benign += (preds == 0).sum().item()  # Predetti come benign
            backdoor_total += len(target)
    
    acc_backdoored = backdoor_correct / backdoor_total
    asr = misclassified_as_benign / backdoor_total  # Attack Success Rate
    
    print(f"\n===  Backdoor Attack Evaluation ===")
    print(f"Acc(F_b, X):   {acc_clean:.4f}  (clean test set - should stay high)")
    print(f"Acc(F_b, X_b): {acc_backdoored:.4f}  (backdoored malware - should drop)")
    print(f"ASR:           {asr:.4f}  (attack success rate - higher is better)")
    print(f"\nInterpretation:")
    print(f"  ✓ Clean acc ~{acc_clean:.2%}: Model still works on normal data")
    print(f"  {'✓' if asr > 0.5 else '✗'} ASR {asr:.2%}: {asr*100:.1f}% of backdoored malware evades detection")
    
    return {
        'acc_clean': float(acc_clean),
        'acc_backdoored': float(acc_backdoored),
        'attack_success_rate': float(asr),
        'backdoored_samples': len(malware_indices)
    }


def compare_with_paper_results(your_results, model_type='EmberNN', trigger_size=8):
    """
    Confronta i tuoi risultati con quelli del paper (Table 3)
    """
    print("\n" + "="*80)
    print(" COMPARISON WITH PAPER RESULTS (Severi et al.)")
    print("="*80)
    
    # Risultati dal paper (Table 6 - LargeAbsSHAP x CountAbsSHAP)
    paper_results = {
        'LightGBM': {
            '4_features': {'poison_1pct': 0.4034, 'poison_3pct': 0.1010},
            '8_features': {'poison_1pct': 0.0282, 'poison_3pct': 0.0104},
            '16_features': {'poison_1pct': 0.0020, 'poison_3pct': 0.0010}
        },
        'EmberNN': {
            '16_features': {'poison_1pct': 0.2104, 'poison_3pct': 0.3676},
            '32_features': {'poison_1pct': 0.1323, 'poison_3pct': 0.2040},
            '128_features': {'poison_1pct': 0.0075, 'poison_3pct': 0.0117}
        }
    }
    
    # Seleziona risultati appropriati
    if model_type in paper_results:
        model_results = paper_results[model_type]
        trigger_key = f'{trigger_size}_features'
        
        if trigger_key in model_results:
            paper_acc_backdoor = model_results[trigger_key]['poison_1pct']
            paper_asr = 1 - paper_acc_backdoor
            
            print(f"\n Paper Results ({model_type}, {trigger_size}-feature trigger, 1% poison):")
            print(f"  Acc(F_b, X_b): {paper_acc_backdoor:.4f}")
            print(f"  ASR:           {paper_asr:.4f} ({paper_asr*100:.1f}%)")
            
            print(f"\n Your Results:")
            print(f"  Acc(F_b, X):   {your_results['acc_clean']:.4f}")
            print(f"  Acc(F_b, X_b): {your_results['acc_backdoored']:.4f}")
            print(f"  ASR:           {your_results['attack_success_rate']:.4f} ({your_results['attack_success_rate']*100:.1f}%)")
            
            # Delta comparison
            delta_asr = your_results['attack_success_rate'] - paper_asr
            delta_acc_backdoor = your_results['acc_backdoored'] - paper_acc_backdoor
            
            print(f"\n Delta (Your - Paper):")
            print(f"  Δ ASR:           {delta_asr:+.4f} ({delta_asr*100:+.1f}%)")
            print(f"  Δ Acc(F_b, X_b): {delta_acc_backdoor:+.4f}")
            
            if delta_asr > 0:
                print(f"  ✓ Your attack is MORE effective!")
            else:
                print(f"  ✗ Your attack is LESS effective")
        else:
            print(f"  No paper results for {trigger_size} features on {model_type}")
    else:
        print(f"  No paper results for {model_type}")
    
    print("\n Note: Exact comparison difficult due to:")
    print("   - Different hardware/implementation")
    print("   - Random seed differences")
    print("   - SHAP approximation variations")
    print("="*80)