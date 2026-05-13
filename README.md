# Predictive Fault Detection in Semiconductor Manufacturing Using Time-Series Deep Learning

## Project Overview

This project focuses on predictive maintenance in semiconductor manufacturing using machine learning and deep learning techniques on multivariate sensor time-series data. The objective is to predict equipment failures before they occur by progressively moving from baseline static models to advanced temporal deep learning architectures.

The project is implemented using the SECOM manufacturing dataset and follows a three-phase modeling pipeline:

1. Baseline Modeling – Logistic Regression
2. Temporal Modeling – LSTM
3. Advanced Early Warning System – Transformer

The study demonstrates how temporal learning and attention-based architectures significantly improve predictive maintenance performance in highly imbalanced industrial environments.

# Problem Statement

Semiconductor manufacturing systems generate large volumes of high-dimensional sensor data from complex fabrication processes. Small deviations in sensor readings can lead to equipment faults, yield loss, and production downtime.

Traditional rule-based monitoring systems are limited because they:

* Cannot effectively capture nonlinear dependencies
* Struggle with temporal relationships
* Perform poorly under noisy high-dimensional conditions

This project aims to build an AI-driven predictive maintenance system capable of:

* Detecting imminent failures
* Learning temporal dependencies
* Providing early warnings before failures occur

# Dataset

## SECOM Manufacturing Dataset

The project uses the SECOM dataset, a widely used benchmark dataset for semiconductor fault detection.

### Dataset Characteristics

* High-dimensional sensor measurements
* Missing sensor readings
* Severe class imbalance
* Binary classification labels
* Real industrial process data

### Challenges

* Missing values
* Noisy sensor signals
* Rare failure events
* Temporal dependency modeling

# Methodology

## Phase 1: Logistic Regression (Baseline)

### Objective

Establish a baseline using static feature vectors without temporal information.

### Preprocessing

* Removed features with excessive missing values
* Mean imputation for remaining missing data
* Feature standardization
* Stratified train-test split

### Imbalance Handling

* Class-weighted BCEWithLogitsLoss
* Threshold tuning using validation F1-score

### Outcome

Provides benchmark performance but struggles with rare failure detection.

## Phase 2: LSTM Temporal Modeling

### Objective

Capture temporal dependencies in sensor data sequences.

### Temporal Feature Engineering

* Rolling means
* Rolling standard deviations
* First-order differences (delta features)

### Sequence Construction

* Fixed-length sequential windows
* Sequence-level labeling

### Model Architecture

* Bidirectional LSTM
* Dropout regularization
* Fully connected output layer

### Training Strategy

* Mini-batch training
* Early stopping
* ROC-AUC monitoring

### Outcome

Improves temporal learning and fault detection but shows instability and overfitting.

## Phase 3: Transformer-Based Early Warning System

### Objective

Predict failures within a future horizon of K time steps.

### Key Improvements

* Self-attention architecture
* Early warning formulation
* Long-range dependency modeling
* Sequence-level representations

### Advantages

* Better temporal learning
* Parallel sequence processing
* Improved generalization
* Strong rare-event detection

### Outcome

Achieves the best overall performance across all evaluation metrics.

# Evaluation Metrics

The models were evaluated using metrics suitable for imbalanced classification problems.

## Metrics Used

### F1-Score

Balances precision and recall for rare-event detection.

### ROC-AUC

Measures ranking capability across thresholds.

### PR-AUC

More informative for highly imbalanced datasets.

### Threshold Optimization

Validation-based threshold tuning was used instead of a fixed 0.5 threshold.

# Results

| Phase | Model               | F1-Score | ROC-AUC | PR-AUC |
| ----- | ------------------- | -------- | ------- | ------ |
| 1     | Logistic Regression | 0.1500   | 0.6370  | 0.1191 |
| 2     | LSTM                | 0.2095   | 0.5937  | 0.1899 |
| 3     | Transformer         | 0.4590   | 0.7519  | 0.4752 |

## Key Observations

* Logistic Regression establishes a baseline but lacks temporal understanding.
* LSTM improves detection performance using sequence learning.
* Transformer achieves the strongest and most stable performance.
* Early warning formulation significantly improves predictive capability.

# References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
2. Vaswani, A., et al. (2017). Attention Is All You Need.
3. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data.
4. Davis, J., & Goadrich, M. (2006). Precision-Recall and ROC Curves.
5. Lee, J., et al. (2016). Introduction to Cyber Manufacturing.
6. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems.

7. Conclusion

This project demonstrates that combining temporal deep learning with an early warning formulation significantly improves predictive maintenance performance in semiconductor manufacturing. Among all approaches, the Transformer-based early warning system provides the best balance of detection capability, ranking performance, and practical industrial applicability.
