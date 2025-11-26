# EMBER Dataset Analysis Report

##  Basic Statistics

- **Training samples**: 600,000
- **Test samples**: 200,000
- **Total features**: 2381
- **Memory size**: 7266.2 MB
- **Train imbalance ratio**: 1.000

##  Feature Quality

- **Constant features**: 40
- **Very sparse (>99% zero)**: 386
- **Features with NaN**: 0
- **Average sparsity**: 72.15%

##  Correlation Analysis

- **Features analyzed**: 2,341
- **High correlation pairs (>0.95)**: 4,712
- **Average correlation**: 0.0307

##  Feature Selection Impact

| Threshold | Removed | Remaining | Reduction % |
|-----------|---------|-----------|-------------|
| 0.90 | 211 | 2,129 | 9.0% |
| 0.95 | 172 | 2,168 | 7.4% |
| 0.98 | 142 | 2,198 | 6.1% |
| 0.99 | 132 | 2,208 | 5.6% |

##  Generated Files

- `correlation_analysis.png` - Correlation heatmap and distribution
- `feature_importance.png` - Mutual Information analysis
- `feature_selection_impact.png` - Impact of correlation thresholds
- `dataset_analysis_report.json` - Complete analysis data
