# SthiraSense-ML

A comprehensive machine learning module for predicting stablecoin depeg events and liquidity stress analysis. This project implements hybrid ensemble models to detect and forecast cryptocurrency stablecoin depeg risks in real-time.

**Built by**: Team NazarBattu

## Overview

SthiraSense-ML provides an advanced ML pipeline for cryptocurrency risk management, specifically focused on stablecoin stability monitoring. It combines multiple machine learning techniques to:

- **Predict depeg events** in 24-hour windows using a hybrid stacked ensemble
- **Calculate continuous liquidity stress scores** (0-100 scale)
- **Estimate time-to-depeg** and confidence intervals
- **Analyze feature importance** for interpretability
- **Evaluate model performance** across different time periods

## Features

### Core ML Capabilities

- **Hybrid Stacked Ensemble**: Combines Random Forest + XGBoost meta-learner architecture
- **Advanced Feature Engineering**: 60+ minute-level technical indicators
- **SMOTE-based Imbalance Handling**: Addresses class imbalance in depeg events
- **Precision-Focused Prediction**: Optimized to reduce false positives
- **Temporal Analysis**: Performance evaluation across rolling time windows
- **Quantile Regression**: Prediction confidence intervals and risk estimation

### Data Processing

- Handles minute-level OHLCV (Open, High, Low, Close, Volume) data
- Rolling window features: 3h, 6h, 12h, 24h, 48h, 7d, 30d
- Automatic feature engineering and normalization
- Missing data handling and outlier detection

## Project Structure

```
SthiraSense-ML/
├── README.md                              # This file
├── LICENSE                                # Project license
├── ML files/
│   ├── train.py                          # Model training pipeline (hybrid ensemble)
│   ├── predict_module.py                 # Inference module for predictions
│   ├── preprocessing.py                  # Feature engineering pipeline
│   ├── liquidity_stress_regression.py    # Continuous stress score modeling
│   ├── importance_analysis.py            # Feature importance analysis
│   ├── temporal_analysis.py              # Temporal model evaluation
│   ├── Model files/                      # Trained model artifacts
│   │   ├── improved_hybrid_depeg_rf.pkl       # Random Forest model
│   │   ├── improved_hybrid_depeg_xgb.pkl      # XGBoost model
│   │   ├── improved_hybrid_depeg_meta.pkl     # Meta-learner model
│   │   └── improved_hybrid_depeg_config.pkl   # Configuration & feature info
│   └── images/                           # Analysis and evaluation plots
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Required Libraries

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib seaborn
```

Or install from requirements (if available):

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing & Feature Engineering

```python
from preprocessing import StablecoinFeatureEngine

# Initialize feature engine
engine = StablecoinFeatureEngine()

# Load your OHLCV data
df = pd.read_csv('raw_stablecoin_data.csv')

# Transform to engineered features
df_engineered = engine.fit_transform(df)
```

**Input Requirements**:
- Columns: `open_time`, `open`, `high`, `low`, `close`, `volume`
- Data frequency: Minute-level OHLCV
- Target column: `target_depeg_next_24h` (binary: 0/1)

### 2. Training Models

```python
from train import ImprovedHybridDepegPredictor

# Initialize and train
predictor = ImprovedHybridDepegPredictor()
X_train, X_test = predictor.prepare_data('processed_data.csv', split_date='2025-11-01')
predictor.train(X_train, y_train)
predictor.save_models('improved_hybrid_depeg')
```

**Model Architecture**:
- **Base Learners**: Random Forest (100 trees) + XGBoost (50 rounds)
- **Meta-Learner**: Logistic Regression with L2 regularization
- **Class Imbalance**: SMOTE oversampling (conservative ratio)
- **Threshold Optimization**: Precision-focused via ROC curves

### 3. Making Predictions

```python
from predict_module import DepegPredictor

# Load trained models
predictor = DepegPredictor()

# Prepare your feature data (use preprocessing.py to engineer features)
features_df = pd.read_csv('features.csv')

# Get depeg predictions
results = predictor.predict(features_df)
# Returns: {'predictions': [...], 'probabilities': [...], 'timestamps': [...]}

# Get risk alerts
alerts = predictor.get_alerts(confidence_threshold=0.7)
```

### 4. Liquidity Stress Analysis

```python
from liquidity_stress_regression import LiquidityStressModel

# Train stress score regressor
stress_model = LiquidityStressModel()
stress_model.train(processed_data)
stress_predictions = stress_model.predict(test_features)

# Stress levels:
# 0-20: Minimal, 20-40: Low, 40-60: Moderate, 60-80: High, 80-100: Critical
```

### 5. Feature Importance Analysis

```python
from importance_analysis import load_and_analyze

# Analyze feature importance across all base models
load_and_analyze('improved_hybrid_depeg', top_n=20)
```

### 6. Temporal Model Evaluation

```python
from temporal_analysis import RealModelTemporalEvaluator

# Evaluate model across different time periods
evaluator = RealModelTemporalEvaluator('processed_data.csv')
evaluator.evaluate_full_period()
evaluator.generate_report()
```

## Model Performance

The trained hybrid ensemble achieves:
- **High Precision**: Minimized false positive depeg alerts
- **Balanced Recall**: Captures majority of actual depeg events
- **Robust Temporal Performance**: Consistent across market cycles
- **Confidence Intervals**: Quantile regression for risk estimation

## Data Format Requirements

### Input CSV Columns
```
open_time, high, low, open, close, volume, market_cap, 
peg_deviation, volume_24h_change, [additional features]
```

### Output Predictions
```python
{
    'timestamp': datetime,
    'depeg_probability': float (0-1),
    'depeg_prediction': int (0 or 1),
    'liquidity_stress_score': float (0-100),
    'confidence': float (0-1),
    'confidence_interval': [lower, upper]
}
```

## Key Modules

### `train.py` (620 lines)
- Hybrid stacked ensemble implementation
- SMOTE imbalance handling
- Threshold optimization for precision
- Cross-validation and performance metrics

### `predict_module.py` (524 lines)
- Production-ready inference pipeline
- Fast batch predictions
- Probability calibration
- Alert generation with confidence thresholds

### `preprocessing.py` (265 lines)
- Minute-level feature engineering
- 7 rolling window sizes (3h-30d)
- Technical indicators and momentum features
- Normalization and scaling

### `liquidity_stress_regression.py` (635 lines)
- Continuous stress score calculation
- Multi-component stress index
- XGBoost + Quantile Regression
- Time-to-depeg estimation

### `importance_analysis.py` (120 lines)
- Feature importance extraction
- RF + XGBoost hybrid scoring
- Visualization and ranking

### `temporal_analysis.py` (611 lines)
- Period-based performance evaluation
- Rolling window assessment
- Metric aggregation and trend analysis

## Configuration

Model parameters are stored in `improved_hybrid_depeg_config.pkl`:
- Feature column names
- Optimal decision threshold
- Scaling parameters
- Training metadata

## Contributing

Team NazarBattu 

## License

[See LICENSE file for details]

## References

- scikit-learn documentation: https://scikit-learn.org
- XGBoost: https://xgboost.readthedocs.io
- Imbalanced-learn (SMOTE): https://imbalanced-learn.org

---

**Last Updated**: February 2026
**Status**: Active Development
