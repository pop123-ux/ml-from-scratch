# Logistic Regression from Scratch

A minimal implementation of **Logistic Regression** trained with batch gradient descent, written from scratch in Python with NumPy — no scikit-learn. Supports **binary** and **multiclass** (one-vs-rest) classification, optional L1/L2 regularization, and ships with a small `ClassificationMetrics` helper (accuracy, precision, recall, F1).

## How it works

Logistic regression predicts the probability that a sample belongs to the positive class by squashing a linear score through the sigmoid function:

1. Compute the linear score `z = X · w + b`.
2. Map it to a probability with the sigmoid `σ(z) = 1 / (1 + e^-z)`.
3. Measure error with **binary cross-entropy** loss.
4. Update weights and bias by gradient descent.
5. Stop early once the change in loss drops below a tolerance.

For **multiclass** problems the model trains one binary classifier per class (**one-vs-rest**) and predicts the class with the highest probability.

## Usage

```python
import numpy as np
from LogisticRegression import LogisticRegression, ClassificationMetrics

# fit() auto-detects binary vs multiclass from the labels
model = LogisticRegression(lr=0.1, n_iters=2000, penalty="l2")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)          # class labels
proba  = model.predict_proba(X_test)    # probabilities

m = ClassificationMetrics(y_test, y_pred)
print(m.accuracy(), m.precision(), m.recall(), m.f1_score())
```

You can also call `fit_binary` / `fit_multiclass` directly if you want to be explicit.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `lr` | Learning rate (default: `0.01`). |
| `n_iters` | Maximum number of gradient descent iterations (default: `1000`). |
| `penalty` | Regularization: `None`, `"l1"` (Lasso), or `"l2"` (Ridge). |
| `tol` | Early-stopping tolerance on the change in loss (default: `1e-5`). |

### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train; auto-routes to binary or one-vs-rest multiclass. |
| `predict(X, threshold=0.5)` | Return predicted class labels. |
| `predict_proba(X)` | Return class probabilities. |

## Files

- `LogisticRegression.py` — the model and `ClassificationMetrics`.
- `ModelsTest.ipynb` — a notebook demonstrating training and evaluation.

## Requirements

- Python 3.x
- NumPy

```bash
pip install numpy
```

## License

MIT
