# ML from Scratch

A growing collection of **machine learning algorithms implemented from scratch** in Python with NumPy — no scikit-learn, no shortcuts. The goal is to understand the theory by building each algorithm by hand, while practicing clean, object-oriented Python.

## Why this repo

Libraries like scikit-learn are great for getting things done, but they hide the math. Re-implementing each algorithm from first principles makes the theory concrete: you see exactly how a model learns, predicts, and where its assumptions live.

## Implementations

| Algorithm | Type | Folder |
|-----------|------|--------|
| K-Nearest Neighbors | Classification | [`knn/`](knn/) |
| Linear Regression | Regression | [`linear-regression/`](linear-regression/) |

Each folder has its own README with the math, usage, and notes.

## Requirements

- Python 3.x
- NumPy

```bash
pip install numpy
```

## Repo layout

```
ml-from-scratch/
├── README.md            # you are here
├── knn/                 # K-Nearest Neighbors
│   ├── KNN.py
│   └── README.md
└── linear-regression/   # Linear Regression (gradient descent + L1/L2)
    ├── LinearRegression.py
    ├── ModelTest.ipynb
    └── README.md
```

## Adding a new algorithm

1. Create a new folder named after the algorithm (lowercase, hyphenated).
2. Put the implementation in a clearly named `.py` file, ideally as a class with `fit` / `predict` methods for consistency.
3. Add a short `README.md` explaining the idea, the math, and how to use it.
4. Add a row to the table above.

## License

MIT
