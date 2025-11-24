import numpy as np

class SimpleNN:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def train(self, X, y, algo="hebb"):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # init weights & bias
        self.w = np.zeros(n_features)
        self.b = 0

        if algo == "hebb":
            # -------------------- HEBB LEARNING --------------------
            for i in range(n_samples):
                self.w += X[i] * y[i]
                self.b += y[i]

        elif algo == "perceptron":
            # -------------------- PERCEPTRON RULE --------------------
            for _ in range(self.epochs):
                for i in range(n_samples):
                    y_pred = np.sign(np.dot(self.w, X[i]) + self.b)
                    if y_pred != y[i]:
                        self.w += self.lr * X[i] * y[i]
                        self.b += self.lr * y[i]

        elif algo == "adaline":
            # -------------------- ADALINE (LMS) --------------------
            for _ in range(self.epochs):
                y_pred = np.dot(X, self.w) + self.b
                error = y - y_pred
                self.w += self.lr * np.dot(error, X)
                self.b += self.lr * np.sum(error)

        else:
            raise ValueError("Choose algo = 'hebb', 'perceptron', or 'adaline'")

    def predict(self, X):
        X = np.array(X)
        return np.sign(np.dot(X, self.w) + self.b)
