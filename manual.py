import numpy as np
from neural_network import NeuralNetwork
from data_Loader import load_mnist

# Load data
_, test_dataset = load_mnist()

# Load trained model
nn = NeuralNetwork()
nn.load("mnist_model.npz")   # uncomment if you saved weights

# Pick any test image
index = int(input("Enter MNIST test image index (0-9999): "))

image, label = test_dataset[index]
X = image.numpy().reshape(1, 784)

output = nn.forward(X)
prediction = np.argmax(output)

print("\nPrediction result")
print("-----------------")
print("Predicted digit:", prediction)
print("Actual digit   :", label)
