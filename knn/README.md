# K-Nearest Neighbors from Scratch

A minimal implementation of the **K-Nearest Neighbors (KNN)** classification algorithm written from scratch in Python with NumPy — no scikit-learn.

## How it works

KNN classifies a sample by looking at the `k` closest points in the training set and taking a majority vote of their labels:

1. Compute the distance from the test sample to every training sample.
2. Pick the `k` nearest neighbors.
3. Assign the most common class label among those neighbors.

## Usage

```python
import numpy as np
from KNN import KNN

clf = KNN(k=5)
y_pred = clf.predict(X_test, X_train, y_train)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `k` | Number of nearest neighbors used to vote on the class (default: `5`). |

## Implementation notes

- `squared_euclidean_distance` — standard Euclidean (L2) distance between two points.
- `manhattan_distance` — Manhattan (L1) distance.
- Voting is done with `np.bincount(...).argmax()` to find the most frequent label.

## Requirements

- Python 3.x
- NumPy

```bash
pip install numpy
```

## License

MIT
