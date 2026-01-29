import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 64) * 0.01
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    # ---------- Activations ----------
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # ---------- Forward ----------
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    # ---------- Backward ----------
    def backward(self, X, y, lr):
        y_onehot = np.zeros((1, 10))
        y_onehot[0, y] = 1

        # Output layer (Softmax + Cross-Entropy)
        d2 = self.a2 - y_onehot

        dW2 = self.a1.T @ d2
        db2 = np.sum(d2, axis=0, keepdims=True)

        # Hidden layer
        d1 = (d2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = X.T @ d1
        db1 = np.sum(d1, axis=0, keepdims=True)

        # Update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    # ---------- Save / Load ----------
    def save(self, path="mnist_model.npz"):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path="mnist_model.npz"):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
