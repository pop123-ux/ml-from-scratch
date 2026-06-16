import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, penalty=None, tol=1e-5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.penalty = penalty
        self.tol = tol
        # set by fit_multiclass when there are more than two classes
        self.classes = None
        self.classifiers = None

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # avoid overflow in exp
        return 1 / (1 + np.exp(-x))

    def _gradient_descent(self, X, y):
        """Train one binary classifier and return (weights, bias)."""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        prev_loss = float('inf')
        for i in range(self.n_iters):
            z = np.dot(X, weights) + bias
            pred = self._sigmoid(z)
            error = pred - y

            # The gradients
            dw = (1 / n_samples) * np.dot(X.T, error)  # w.r.t. the weights
            db = (1 / n_samples) * np.sum(error)       # w.r.t. the bias

            # regularization derivative term
            if self.penalty == 'l1':
                dw += self.lr * np.sign(weights)
            elif self.penalty == 'l2':
                dw += self.lr * 2 * weights

            weights = weights - (self.lr * dw)
            bias = bias - (self.lr * db)

            pred_clipped = np.clip(pred, 1e-15, 1 - 1e-15)  # prevent log(0)
            current_loss = -(1 / n_samples) * np.sum(
                y * np.log(pred_clipped) + (1 - y) * np.log(1 - pred_clipped)
            )

            if abs(current_loss - prev_loss) < self.tol:  # early stopping
                break

            prev_loss = current_loss

        return weights, bias

    def fit(self, X, y):
        """Auto-route: binary if <=2 classes, otherwise one-vs-rest."""
        if len(np.unique(y)) > 2:
            self.fit_multiclass(X, y)
        else:
            self.fit_binary(X, y)
        return self

    def fit_binary(self, X, y):
        self.weights, self.bias = self._gradient_descent(X, y)
        return self

    def fit_multiclass(self, X, y):
        self.classes = np.unique(y)  # how many unique classes we have
        self.classifiers = []

        for c in self.classes:
            # one-vs-rest: 1 for the current class, 0 for everything else
            y_binary = np.where(y == c, 1, 0)
            w_c, b_c = self._gradient_descent(X, y_binary)
            self.classifiers.append((w_c, b_c))

        return self

    def predict_proba(self, X):
        if self.classifiers is not None:
            # one column of probabilities per class
            all_probabilities = []
            for w_c, b_c in self.classifiers:
                z = np.dot(X, w_c) + b_c
                all_probabilities.append(self._sigmoid(z))
            return np.column_stack(all_probabilities)

        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        if self.classifiers is not None:
            # index of the class with the highest probability, mapped back to its label
            return self.classes[np.argmax(probabilities, axis=1)]
        return np.where(probabilities >= threshold, 1, 0)


class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tp, self.tn, self.fp, self.fn = self._get_counts()

    def _get_counts(self):
        tp, tn, fp, fn = 0, 0, 0, 0
        for true, pred in zip(self.y_true, self.y_pred):
            if true == 1 and pred == 1:
                tp += 1
            if true == 1 and pred == 0:
                fn += 1
            if true == 0 and pred == 1:
                fp += 1
            if true == 0 and pred == 0:
                tn += 1
        return tp, tn, fp, fn

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    def precision(self):
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    def recall(self):
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if (p + r) == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
