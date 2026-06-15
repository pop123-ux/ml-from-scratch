# Linear Regression from Scratch

A minimal implementation of **Linear Regression** trained with batch gradient descent, written from scratch in Python with NumPy — no scikit-learn. Supports optional L1 (Lasso) and L2 (Ridge) regularization.

## How it works

The model fits weights and a bias by minimizing mean squared error via gradient descent:

1. Predict `y = X · w + b`.
2. Compute gradients of the loss w.r.t. weights and bias.
3. Update parameters by the learning rate.
4. Stop early once the change in loss drops below a tolerance.

## Usage

```python
import numpy as np
from LinearRegression import LinearRegression

model = LinearRegression(lr=0.001, n_iters=1000, penalty="l2")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `lr` | Learning rate (default: `0.001`). |
| `n_iters` | Maximum number of gradient descent iterations (default: `1000`). |
| `penalty` | Regularization: `None`, `"l1"` (Lasso), or `"l2"` (Ridge). |
| `tol` | Early-stopping tolerance on the change in loss (default: `1e-5`). |

## Files

- `LinearRegression.py` — the model implementation.
- `ModelTest.ipynb` — a notebook demonstrating training and evaluation.

## Requirements

- Python 3.x
- NumPy

```bash
pip install numpy
```

## License

MIT
