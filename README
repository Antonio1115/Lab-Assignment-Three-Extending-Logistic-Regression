# One-vs-All and One-vs-One Logistic Regression Classifier

## Overview
This project implements a **custom logistic regression classifier** in Python using **NumPy**, following a scikit-learn style object-oriented design. The classifier supports:

- Binary and multi-class classification
- Multiple optimization methods:
  - Steepest ascent
  - Stochastic gradient ascent
  - Newton’s method
- Regularization options:
  - L1 (Lasso)
  - L2 (Ridge)
  - L1 + L2 (Elastic Net)
- Multi-class strategies:
  - One-vs-All (OvA)
  - One-vs-One (OvO)

The purpose of this project is to explore **optimization techniques**, **regularization effects**, and **performance trade-offs** compared to scikit-learn’s logistic regression.

---

## Features

### Binary Logistic Regression
- Base class supports probability prediction, intercept addition, and customizable regularization.
- Optimized with three solvers:
  - **Steepest ascent** (batch gradient)
  - **Stochastic gradient ascent** (mini-batches)
  - **Newton’s method** (second-order optimization)
- Flexible regularization strength controlled via `C`.

### Multi-class Logistic Regression
- **One-vs-All (OvA) approach:** trains a separate binary classifier for each class.
- Predicts the class with the **highest probability**.
- Supports all solvers and regularization options.

### One-vs-One Logistic Regression
- Trains a binary classifier for **every pair of classes**.
- Uses a **voting mechanism** to determine final prediction.
- Often improves accuracy on **complex or imbalanced datasets**.

---

## Implementation Highlights
- **Numerical stability:** sigmoid function clipped to prevent overflow.
- **Vectorized operations:** gradient and Hessian calculations optimized using NumPy.
- **Cross-validation:** K-Fold validation used to tune hyperparameters `C`, learning rate `η`, and iteration count.
- **Comparison with scikit-learn:** demonstrated trade-offs between speed and accuracy.

---

## Hyperparameter Tuning & Results

### L2 Regularization
| Solver      | Best C | Avg Accuracy |
|------------|--------|-------------|
| Steepest   | 0.001  | ~80%        |
| Stochastic | 0.001  | ~80%        |
| Newton     | 0.0032 | ~80%        |

### L1 Regularization
| Solver      | Best C | Avg Accuracy |
|------------|--------|-------------|
| Steepest   | 0.001  | ~80%        |
| Stochastic | 0.01   | ~82%        |
| Newton     | 0.001  | 92.98%      |

### L1 + L2 (Elastic Net)
| Solver      | Best C | Avg Accuracy |
|------------|--------|-------------|
| Steepest   | 0.001  | ~80%        |
| Stochastic | 0.001  | ~80%        |
| Newton     | 0.001  | ~82%        |

> Newton’s method with **L1 regularization (C=0.001)** achieved the **highest performance**.

---

## Custom vs Scikit-Learn Comparison
| Model                       | Accuracy | Execution Time (sec) |
|------------------------------|---------|--------------------|
| Custom Logistic Regression   | 92.98%  | 17.92              |
| Scikit-Learn Logistic Reg.   | 99.11%  | 140.61             |

**Insight:**  
- Custom model is **faster** but slightly less accurate.  
- Scikit-learn is **more accurate**, using advanced optimization methods but takes longer.

---

## One-vs-One Classifier
- Custom One-vs-One Logistic Regression achieved **89.18% accuracy** in **6.02 sec**.
- This approach breaks multi-class classification into pairwise comparisons, improving robustness in complex or imbalanced datasets.

---

## Key Insights
- Newton’s method with L1 regularization produced the **highest accuracy**.
- Regularization strength `C` significantly impacts performance; smaller values prevented overfitting in most cases.
- Custom implementations offer **speed advantages**, while scikit-learn provides **higher accuracy** due to optimized solvers.
- One-vs-One strategy is effective for **imbalanced or multi-class datasets** but increases computational cost.

---

## Requirements
- Python 3.x
- NumPy
- SciPy
- scikit-learn
- matplotlib

Install dependencies via pip:
```bash
pip install numpy scipy scikit-learn matplotlib
