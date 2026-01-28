import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 64) * 0.01
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, lr):
        y_onehot = np.zeros((1, 10))
        y_onehot[0, y] = 1

        error = self.a2 - y_onehot
        d2 = error * self.sigmoid_derivative(self.a2)

        dW2 = self.a1.T @ d2
        db2 = d2

        d1 = (d2 @ self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = X.T @ d1
        db1 = d1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
