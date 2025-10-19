1. Purpose

The objective of this project is to predict the overall car evaluation category using features such as buying price, maintenance cost, number of doors, capacity, luggage boot size, and safety.

Goal: Develop and compare multiple supervised machine learning models to determine which provides the best predictive performance.

Models tested include:

Logistic Regression

Decision Tree

Neural Network (MLP)

GridSearchCV optimized variants of each model

This study demonstrates data-driven classification for automotive decision-making, useful for customers, dealers, and automated evaluation systems.



2. Methodology
2.1 Dataset

Dataset obtained from the UCI Car Evaluation dataset.

Features include categorical and ordinal variables:

buying, maint, doors, persons, lug_boot, safety

Target variable: class (car acceptability: unacc, acc, good, vgood)

2.2 Data Preprocessing

Categorical features encoded using Label Encoding / One-Hot Encoding as appropriate.

Dataset split into training and testing sets.

Standardization/scaling applied for models sensitive to feature scale (e.g., MLP).

2.3 Models and Training

Logistic Regression (LR) – baseline linear classifier.

Decision Tree (DT) – tree-based classifier for interpretable rules.

Neural Network (MLP) – feed-forward multi-layer perceptron with hidden layers.

GridSearchCV (GS) – hyperparameter optimization for LR, DT, and MLP.

Evaluation metrics:

Accuracy – overall prediction correctness

Cross-Validation Accuracy (CV Accuracy) – mean accuracy from k-fold CV

Mean Squared Error (MSE) – squared difference between predicted and true class labels

Mean Absolute Error (MAE) – absolute difference between predicted and true class labels


RESULTS:
| Model                               | Accuracy | CV Accuracy | MSE   | MAE   |
| ----------------------------------- | -------- | ----------- | ----- | ----- |
| Logistic Regression                 | 0.830    | 0.820       | 0.268 | 0.198 |
| GridSearchCV + Logistic Regression  | 0.834    | 0.825       | 0.264 | 0.195 |
| Decision Tree                       | 0.871    | 0.861       | 0.193 | 0.150 |
| GridSearchCV + Decision Tree        | 0.965    | 0.983       | 0.046 | 0.039 |
| Neural Network (MLP)                | 0.911    | 0.924       | 0.123 | 0.100 |
| GridSearchCV + Neural Network (MLP) | 0.990    | 0.991       | 0.015 | 0.012 |
