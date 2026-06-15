import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000, penalty=None, tol=1e-5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.penalty = penalty
        self.tol = tol
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        prev_loss = 0
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            # Cost function
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)))
            # regularization derivative term
            if self.penalty == 'l1':
                dw += self.lr * 2 * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += self.lr * 2 * self.weights

            db = (1/n_samples) * (np.sum(y_pred - y))
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

            current_loss = np.mean(np.square(y_pred - y))

            if abs(current_loss - prev_loss) < self.tol: # tolerance
                break

            prev_loss = current_loss

    def predict(self, pred):
        return np.dot(pred, self.weights) + self.bias
        

    