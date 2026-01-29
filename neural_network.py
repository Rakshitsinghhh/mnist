import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 64) * 0.01
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, a):
        return a * (1 - a)   # derivative wrt activation

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, lr):
        # one-hot label
        y_onehot = np.zeros((1, 10))
        y_onehot[0, y] = 1

        # ---- output layer ----
        error = self.a2 - y_onehot
        d2 = error * self.sigmoid_derivative(self.a2)

        dW2 = self.a1.T @ d2
        db2 = np.sum(d2, axis=0, keepdims=True)

        # ---- hidden layer ----
        d1 = (d2 @ self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = X.T @ d1
        db1 = np.sum(d1, axis=0, keepdims=True)

        # ---- update ----
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    # ---------- SAVE / LOAD ----------
    def save(self, path="mnist_model.npz"):
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2
        )

    def load(self, path="mnist_model.npz"):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
