# EMBER Backdoor Detection - Complete Project

## Project Overview

This project implements a comprehensive backdoor attack and defense system for malware classifiers, specifically targeting the EMBER dataset. It reproduces and extends the research from "Explanation-Guided Backdoor Poisoning Attacks Against Malware Classifiers" (Severi et al., USENIX Security 2021).

### Key Features

- **SHAP-Guided Backdoor Attack**: Clean-label backdoor using explainability-guided trigger selection
- **Multiple Defense Strategies**: Isolation Forest, Weight Pruning, Gaussian Noise
- **Complete Pipeline**: From dataset preparation to attack evaluation and defense comparison
- **Comprehensive Metrics**: Attack success rate, model performance, defense effectiveness

---

## Project Structure

```
.
├── Code/
│   ├── main.py                          # Main experiment orchestrator
│   ├── attack/
│   │   ├── backdoor_attack.py          # SHAP-guided backdoor implementation
│   │   ├── poisoning.py                # Label flipping attacks
│   │   ├── poisoning_detector.py       # Noise-based detection
│   │   └── pruning_detector.py         # Pruning-based detection
│   ├── defense/
│   │   └── isolation_forest_detector.py # Baseline defense from paper
│   ├── network/
│   │   ├── model.py                    # EmberNN architecture
│   │   └── trainer.py                  # Training pipeline
│   ├── preprocessing/
│   │   ├── data_loader.py              # EMBER data loading
│   │   ├── vectorize_ember.py          # Dataset vectorization script
│   │   └── diagnostic.py               # Dataset validation tool
│   ├── utils/
│   │   ├── metrics.py                  # Performance metrics
│   │   ├── visualization.py            # Basic plotting functions
│   │   ├── result_analysis.py          # Advanced cross-experiment analysis
│   │   ├── gaussian_noise_analysis.py  # Gaussian noise defense analysis
│   │   ├── pruning_analysis.py         # Pruning defense analysis
│   │   ├── dataset_analysis.py         # Dataset quality analysis
│   │   ├── backdoor_metrics.py         # Attack-specific metrics
│   │   └── io_utils.py                 # I/O utilities
│   └── experiment/
│       └── experiment.py               # Experiment workflows
│
├── dataset/
│   └── ember_dataset_2018_2/           # EMBER dataset (created by setup)
│
├── Repository/                          # Git repositories (created by setup)
│   ├── ember/                          # Official EMBER repo
│   └── MalwareBackdoors/               # Poisoning reference code
│
├── Bibliography/                        # All papers used for the project
│     
├── Documentation/                       # Latex files and PDF documentation
│   
├── Results/                             # All results from different runs
│   ├── ember2018 - mac/                # Mac environment results
│   │   ├── poison rate 1%              
│   │   │   ├── triggersize16           
│   │   │   │   ├── backdoor_experiment_results.json
│   │   │   │   ├── backdoor_comparison_plot.png
│   │   │   │   ├── pruning_analysis_with_baseline.png
│   │   │   │   └── gaussian_noise_analysis.png
│   │   │   ├── triggersize32
│   │   │   ├── triggersize48
│   │   │   ├── triggersize64
│   │   │   └── triggersize128
│   │   ├── poison rate 3%              
│   │   │   ├── triggersize16 - 128 (same structure as above)
│   │   │   └── ...
│   │   └── analysis_plots/             # Cross-experiment comparison plots
│   │       ├── summary_table.csv
│   │       ├── comprehensive_results.csv
│   │       ├── defense_comparison_table.csv
│   │       ├── best_configurations.csv
│   │       ├── 1_danger_heatmap.png
│   │       ├── 2_stealthiness.png
│   │       ├── 3_defense_variation_accuracy.png
│   │       ├── 4_tradeoff_final.png
│   │       ├── 5-14_[various_analysis].png
│   │       ├── pruning_trigger_size_comparison_with_baseline.png
│   │       ├── pruning_poison_rate_comparison_with_baseline.png
│   │       ├── gaussian_noise_comparative_analysis.png
│   │       └── poison_rate_comparison_trigger*.png
│   │   
│   │
│   └── ember2018 - cluster/            # Cluster environment results
│   |   └── [same structure as mac]
│   │ 
|   └── dataset_analysis/           # Dataset quality analysis
│           ├── dataset_analysis_report.json
│           ├── ANALYSIS_SUMMARY.md
│           ├── correlation_analysis.png
│           ├── feature_importance.png
│           └── feature_selection_impact.png
|
├── Presentation/                        # PowerPoint presentation
│     
├── setup.sh                            # Complete setup script
├── activate_ember.sh                   # Environment activation
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Quick Start Guide

### Step 1: Initial Setup

Run the setup script to create the environment and download datasets:

```bash
chmod +x setup.sh
./setup.sh
```

**What this does:**
- Creates `ember_env` conda environment with Python 3.9
- Installs all required dependencies (PyTorch, scikit-learn, SHAP, etc.)
- Downloads EMBER datasets to `dataset/` directory
- Clones necessary repositories to `Repository/`
- Creates activation script for convenience

**Interactive prompts:**
- Download datasets now? (Y/N)
- Extract datasets now? (Y/N)
- Recreate environment if exists? (Y/N)

### Step 2: Vectorize Dataset

After setup, convert the JSONL files to vectorized format:

```bash
# Activate environment
source activate_ember.sh
# or
conda activate ember_env

# Navigate to code directory
cd Code

# Vectorize the dataset
python preprocessing/vectorize_ember.py ../dataset/ember_dataset_2018_2
```

**Expected output:**
```
EMBER Dataset Vectorization
============================
✓ train_features_0.jsonl
✓ train_features_1.jsonl
...
✓ test_features.jsonl

[Processing] This will take 10-30 minutes...

✓ VECTORIZATION COMPLETED!
  ✓ X_train.dat (6,847.2 MB)
  ✓ y_train.dat (6.9 MB)
  ✓ X_test.dat (1,521.3 MB)
  ✓ y_test.dat (1.5 MB)
```

**Note:** Vectorization is computationally intensive and requires:
- ~10-30 minutes on modern hardware
- ~10 GB disk space for vectorized files
- ~16 GB RAM recommended

### Step 3: Run Main Experiment

Execute the complete backdoor attack and defense pipeline:

```bash
python main.py ../dataset/ember_dataset_2018_2
```

**What this does:**
1. **Phase 1**: Trains clean baseline model
2. **Phase 2**: Executes SHAP-guided backdoor attack
3. **Phase 3**: Detects poisoned samples (Weight Pruning)
4. **Phase 4**: Applies Isolation Forest defense (paper baseline)
5. **Phase 5**: Applies Weight Pruning defense
6. **Phase 6**: Applies Gaussian Noise defense

**Interactive prompts during execution:**
```
Eseguire Isolation Forest defense (paper baseline)? (Y/n):
Eseguire Weight Pruning defense (your method)? (Y/n):
Eseguire Gaussian Noise defense (your method)? (Y/n):
```

### Step 4: Dataset Quality Analysis
Analyze the EMBER dataset quality and characteristics:
```bash
cd Code/utils
python dataset_analysis.py ../../dataset/ember_dataset_2018_2
```

**What this does:**
- Basic statistics (samples, features, class distribution)
- Feature quality analysis (constant, sparse, NaN features)
- Correlation analysis with heatmaps
- Feature importance via Mutual Information
- Feature selection impact simulation
- Generates comprehensive report and visualizations

**Output location:**
```
Results/ember2018/dataset_analysis/
  ├── dataset_analysis_report.json      # Complete analysis data
  ├── ANALYSIS_SUMMARY.md               # Human-readable summary
  ├── correlation_analysis.png          # Correlation heatmap
  ├── feature_importance.png            # MI analysis
  └── feature_selection_impact.png      # Selection simulation
```

### Step 5: Cross-Experiment Analysis
After running multiple experiments with different configurations, generate comparative analysis:
```bash
cd Code/utils
python result_analysis.py
```

**What this does:**
- Loads all experiment results from `Results/` directory
- Generates comprehensive comparison tables (CSV)
- Creates 14+ advanced visualization plots
- Identifies best configurations
- Performs statistical analysis

**Output location:**
```
Results/ember2018 - [mac|cluster]/analysis_plots/
  ├── summary_table.csv                 # Quick overview
  ├── comprehensive_results.csv         # Full metrics with deltas
  ├── defense_comparison_table.csv      # Defense effectiveness
  ├── best_configurations.csv           # Optimal configs
  ├── 1_danger_heatmap.png             # Real attack danger
  ├── 2_stealthiness.png               # Attack stealth analysis
  ├── 3_defense_variation_*.png        # Defense effectiveness
  ├── 4_tradeoff_final.png             # Attack trade-offs
  ├── 5-14_*.png                       # Additional analyses
  └── [defense-specific plots]
```

### Step 6: Defense-Specific Analysis
Gaussian Noise Analysis
``` bash
cd Code/utils
python gaussian_noise_analysis.py
```

Generates:
- Individual plots for each configuration (in experiment folders)
- Comparative plots across trigger sizes
- Poison rate comparison (1% vs 3%)
- Saved in respective triggersize* folders and analysis_plots/

### Step 7: Pruning Defense Analysis
```bash
cd Code/utils
python pruning_analysis.py
```
Generates:
- Individual pruning analysis with baseline comparison
- Trigger size comparison plots
- Poison rate comparison plots
- All saved in respective folders and analysis_plots/



---

## Configuration & Parameters

### Main Configuration

Edit `ExperimentConfig` class in `main.py`:

```python
class ExperimentConfig:
    # Data
    DATA_DIR = "dataset/ember_dataset_2018_2"
    
    # Training
    EPOCHS = 10                    # Training epochs
    BATCH_SIZE = 256               # Batch size
    LEARNING_RATE = 0.001          # Learning rate
    DROPOUT_RATE = 0.5             # Dropout rate
    WEIGHT_DECAY = 1e-5            # L2 regularization
    
    # Backdoor Attack
    POISON_RATE = 0.03             # 3% poisoning rate (1% or 3% in paper)
    TRIGGER_SIZE = 64              # Trigger features (8, 16, 32, 64, 128)
    ATTACK_TYPE = 'Clean-Label Backdoor (SHAP-guided)'
    
    # Detection (Weight Pruning)
    DETECTION_PRUNING_RATES = [0.0, 0.05, 0.1, ..., 0.9]
    
    # Defense Parameters
    DEFENSE_PRUNING_RATE = None    # Auto-tuned
    NOISE_STD = 0.03               # Gaussian noise std
    
    # Feature Selection
    CORR_THRESHOLD = 0.98          # Correlation threshold
    MI_TOP_K = None                # Mutual info top-k (None = use all after corr)
```

### Command Line Usage

```bash
# Basic usage (uses default dataset path in config)
python main.py

# Specify custom dataset path
python main.py /path/to/ember/dataset

# With environment activation
source activate_ember.sh && cd Code && python main.py
```

### Key Parameters Explained

**Backdoor Attack:**
- `POISON_RATE`: Fraction of benign samples to poison (paper uses 0.01 or 0.03)
- `TRIGGER_SIZE`: Number of features in trigger (paper tests 8-128 for EmberNN)
- Clean-label: Poisoned samples keep label=0 (benign) for stealthiness

**Detection:**
- `DETECTION_PRUNING_RATES`: Progressive pruning levels to test stability
- Higher stability = clean sample; Lower stability = poisoned sample

**Defense:**
- `DEFENSE_PRUNING_RATE`: Auto-tuned to balance accuracy and robustness
- `NOISE_STD`: Gaussian noise for weight perturbation (auto-tuned)

---

## Expected Output & Results

### Generated Files

After running `main.py`, the following files are created:

**Models (PyTorch checkpoints):**
```
model_clean.pth                    # Baseline model
model_backdoored.pth               # Backdoored model
model_isolation_forest_defended.pth # Defended with IsoForest
model_pruned.pth                   # Defended with pruning
model_noisy.pth                    # Defended with noise
```

**Attack Artifacts:**
```
backdoor_trigger.npy               # Trigger pattern (features + values)
poison_indices_backdoor.npy        # Indices of poisoned samples
```

**Results & Visualizations:**
```
backdoor_experiment_results.json   # Complete metrics (JSON)
backdoor_comparison_plot.png       # 4-panel comparison
backdoor_comparison_enhanced.png   # 5-column detailed comparison
pruning_detection_results.png      # Detection analysis
isolation_forest_detection.png     # IsoForest detection
experiment_config.json             # Saved configuration
selected_features.json             # Feature selection info
```

### Cross-Experiment Analysis Files

Generated in `Results/ember2018 - [environment]/analysis_plots/`:

**CSV Tables:**
```
summary_table.csv                  # Quick metrics overview
comprehensive_results.csv          # Full metrics with Δ and Variation%
defense_comparison_table.csv       # Defense effectiveness comparison
best_configurations.csv            # Optimal configuration recommendations
```

**Visualization Categories:**

1. **Attack Analysis (Plots 1-4):**
   - `1_danger_heatmap.png`: Real backdoor danger by configuration
   - `2_stealthiness.png`: Attack stealth (accuracy change)
   - `2bis_stealthiness_f1.png`: F1-score stealth analysis
   - `4_tradeoff_final.png`: Stealth vs danger trade-off

2. **Defense Analysis (Plots 3, 5-8):**
   - `3_defense_variation_accuracy.png`: Defense vs backdoor (accuracy)
   - `3bis_defense_variation_f1.png`: Defense vs backdoor (F1)
   - `5_f1_comparison.png`: F1 across all models
   - `6_metrics_heatmaps.png`: Multi-metric heatmaps
   - `8_defense_accuracy_detailed.png`: Detailed defense comparison
   - `8bis_defense_f1_detailed.png`: F1 defense comparison

3. **Advanced Analysis (Plots 7-14):**
   - `7_asr_by_config.png`: Attack success rate trends
   - `9_attack_effectiveness_quadrant.png`: ASR vs accuracy drop
   - `10_metrics_by_triggersize.png`: Metrics by trigger size
   - `11_roc_curves_comparison.png`: ROC analysis
   - `12_boxplots_by_poison_rate.png`: Distribution by poison rate
   - `13_boxplots_by_trigger_size.png`: Distribution by trigger size
   - `14_violin_defense_recovery.png`: Recovery distribution

4. **Defense-Specific Plots:**
   - `pruning_trigger_size_comparison_with_baseline.png`
   - `pruning_poison_rate_comparison_with_baseline.png`
   - `gaussian_noise_comparative_analysis.png`
   - `poison_rate_comparison_trigger[16|32|48|64|128].png`

### Dataset Analysis Files

Generated in `Results/ember2018/dataset_analysis/`:
```
dataset_analysis_report.json       # Complete analysis (JSON)
ANALYSIS_SUMMARY.md                # Human-readable summary
correlation_analysis.png           # Correlation heatmap + distribution
feature_importance.png             # MI scores + cumulative importance
feature_selection_impact.png       # Impact of different thresholds
```

### Key Metrics

**1. Attack Effectiveness (Phase 2):**
```
=== Backdoor Attack Evaluation ===
Acc(F_b, X):   0.9945  (clean test set - should stay high)
Acc(F_b, X_b): 0.2104  (backdoored malware - should drop)
ASR:           0.7896  (attack success rate - higher is better)

Interpretation:
  ✓ Clean acc ~99.45%: Model still works on normal data
  ✓ ASR 78.96%: 78.9% of backdoored malware evades detection
```

**2. Defense Effectiveness:**
```
DEFENSE COMPARISON (vs Paper Baseline)
========================================================================
Method                         Accuracy     Recovery     FP Rate
========================================================================
Paper: Isolation Forest        ~0.992       ~99%         11.2%
Paper: Spectral Signatures     ~0.712       ~30-70%      45.0%
------------------------------------------------------------------------
Ours: Isolation Forest         0.9923       97.8%        10.5%
Ours: Weight Pruning           0.9887       89.3%        N/A
Ours: Gaussian Noise           0.9854       82.1%        N/A
========================================================================
```

### Interpretation Guide

**Attack Success Rate (ASR):**
- ASR > 70%: Highly effective attack
- ASR 40-70%: Moderately effective
- ASR < 40%: Weak attack

**Defense Recovery:**
- Recovery = (Acc_defended - Acc_poisoned) / (Acc_clean - Acc_poisoned) × 100%
- Recovery > 90%: Excellent defense
- Recovery 70-90%: Good defense
- Recovery < 70%: Limited effectiveness

---

## Advanced Features

### Feature Selection

The pipeline includes intelligent feature selection:

```python
# Automatic correlation-based removal
select_features(
    X_train, y_train, X_test,
    corr_threshold=0.98,  # Remove features with >98% correlation
    mi_top_k=500,         # Keep top-500 by mutual information
    sample_size=10000     # Use subset for speed
)
```

**Memory optimization for Mac/ARM:**
- Subsampling for correlation computation
- Parallel processing with joblib
- Reduced memory footprint

### Low Memory Mode

For systems with limited RAM (e.g., MacBook Air):

```python
# In experiment.py
experiment_isolation_forest_defense(
    ...,
    low_memory_mode=True  # Reduces sample sizes
)

# Manual control
defender.fit_detector(
    X_train, y_train,
    max_samples_mi=30000,      # Reduced from 50k
    max_samples_forest=50000   # Reduced from 100k
)
```

### Threshold Optimization

Models automatically find optimal classification threshold:

```python
# In trainer.py
optimal_threshold, optimal_f1 = MetricsCalculator.find_optimal_threshold(
    targets, probs, metric='f1'
)

# Recalculate metrics with optimal threshold
preds_optimal = (probs > optimal_threshold).astype(int)
```

---

## Visualization & Analysis

### Generated Plots

**1. Backdoor Comparison Enhanced (5-column):**
- Column 1: F1-Score (primary metric)
- Column 2: Accuracy
- Column 3: Precision
- Column 4: Recall
- Column 5: Attack Success Rate (ASR)
- Bottom row: Confusion matrices for all models

**2. Pruning Detection Results:**
- Stability score distribution
- Sample-wise stability scatter
- Pruning impact curve
- Detection confusion matrix
- Clean vs. Poisoned comparison

**3. Isolation Forest Detection:**
- Outlier score distribution
- Sample-wise anomaly scatter
- Detection metrics bar chart

### Custom Analysis

```python
# Load results
import json
with open('backdoor_experiment_results.json') as f:
    results = json.load(f)

# Extract metrics
clean_acc = results['clean']['test']['accuracy']
backdoor_asr = results['backdoored']['attack_metrics']['attack_success_rate']

# Custom plotting
from utils.visualization import plot_comparison_enhanced
plot_comparison_enhanced(results, save_path='custom_plot.png')
```

---

## Troubleshooting

### Common Issues

**1. SHAP Memory Error:**
```
RuntimeError: CUDA out of memory / OOM
```
**Solution:** Use reduced sample sizes:
```python
backdoor.select_trigger_features_shap_efficient(
    model, X_train, y_train, device,
    sample_size=50,        # Reduced from 100
    background_size=25     # Reduced from 50
)
```

**2. Feature Dimension Mismatch:**
```
ValueError: X_test has 2381 features, but model expects 500
```
**Solution:** Ensure same feature selection:
```bash
# Delete and regenerate
rm selected_features.json
rm model_*.pth
python main.py  # Will recreate with consistent features
```

**3. Slow Vectorization:**
```
[Processing] This will take 10-30 minutes...
```
**Expected behavior.** To speed up:
- Use SSD instead of HDD
- Close other memory-intensive applications
- Consider using pre-vectorized dataset if available

**4. Conda Environment Issues:**
```
[ERRORE] Impossibile attivare l'ambiente ember_env
```
**Solution:**
```bash
# Reinitialize conda
conda init bash
source ~/.bashrc

# Recreate environment
conda env remove --name ember_env
./setup.sh  # Select Y to recreate
```

### Performance Optimization

**For faster experimentation:**

```python
# Reduce epochs
config.EPOCHS = 5  # Instead of 10

# Use smaller dataset subset (for testing only)
X_train = X_train[:50000]
y_train = y_train[:50000]

# Skip optional defenses
# When prompted, answer 'n' to skip

# Reduce pruning detection levels
config.DETECTION_PRUNING_RATES = [0.0, 0.2, 0.5, 0.8]  # Instead of 13 levels
```

---

## References & Resources

### Papers

1. **Severi et al. (2021)**  
   *"Explanation-Guided Backdoor Poisoning Attacks Against Malware Classifiers"*  
   USENIX Security Symposium  
   [Paper Link](https://arxiv.org/abs/2003.01031)

2. **Anderson & Roth (2018)**  
   *"EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models"*  
   [Paper Link](https://arxiv.org/abs/1804.04637)

### Repositories

- **EMBER Official**: https://github.com/elastic/ember
- **MalwareBackdoors**: https://github.com/ClonedOne/MalwareBackdoors
- **SHAP Library**: https://github.com/slundberg/shap

### Datasets

- **EMBER 2017**: https://ember.elastic.co/ember_dataset.tar.bz2
- **EMBER 2017 v2**: https://ember.elastic.co/ember_dataset_2017_2.tar.bz2
- **EMBER 2018 v2**: https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

### Documentation

- EMBER Features: 2381 features extracted from PE files
- Architecture: EmberNN (4000 → 2000 → 100 → 1)
- SHAP: SHapley Additive exPlanations for model interpretability

---

## License & Citation

This project is for **academic and research purposes only**.

### Citation

If you use this code, please cite:

```bibtex
@inproceedings{severi2021explanation,
  title={Explanation-guided backdoor poisoning attacks against malware classifiers},
  author={Severi, Giorgio and Meyer, Jim and Coull, Scott and Oprea, Alina},
  booktitle={30th USENIX Security Symposium},
  pages={1487--1504},
  year={2021}
}

@article{anderson2018ember,
  title={Ember: An open dataset for training static pe malware machine learning models},
  author={Anderson, Hyrum S and Roth, Phil},
  journal={arXiv preprint arXiv:1804.04637},
  year={2018}
}
```

---

## Support

For issues and questions:

1. Check **Troubleshooting** section above
2. Review generated logs in console output
3. Examine `backdoor_experiment_results.json` for detailed metrics
4. Verify dataset integrity with `python preprocessing/diagnostic.py`

---

# Contrinutor:
- Daniele Russo - mat.0001186664
- Nicola Modugno - mat.0001176883

**Last Updated**: 11 December 2025
**Compatible With**: EMBER 2018 v2, Python 3.9+, PyTorch 1.9+