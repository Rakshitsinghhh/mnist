import numpy as np
from neural_network import NeuralNetwork
from data_Loader import load_mnist

# -------------------------------
# Hyperparameters
# -------------------------------
EPOCHS = 3
LEARNING_RATE = 0.01

# -------------------------------
# Load data
# -------------------------------
print("Loading MNIST dataset...")
train_dataset, test_dataset = load_mnist()
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# -------------------------------
# Initialize network
# -------------------------------
nn = NeuralNetwork()

# -------------------------------
# Training loop
# -------------------------------
print("\nStarting training...\n")

for epoch in range(EPOCHS):
    correct = 0

    for image, label in train_dataset:
        # Flatten image (28x28 -> 784)
        X = image.numpy().reshape(1, 784)
        y = label

        # Forward pass
        output = nn.forward(X)
        prediction = np.argmax(output)

        if prediction == y:
            correct += 1

        # Backward pass (learning)
        nn.backward(X, y, lr=LEARNING_RATE)

    accuracy = (correct / len(train_dataset)) * 100
    print(f"Epoch {epoch + 1}/{EPOCHS} - Training Accuracy: {accuracy:.2f}%")

# -------------------------------
# Testing loop
# -------------------------------
print("\nTesting on test dataset...\n")

correct = 0
for image, label in test_dataset:
    X = image.numpy().reshape(1, 784)
    output = nn.forward(X)
    prediction = np.argmax(output)

    if prediction == label:
        correct += 1

test_accuracy = (correct / len(test_dataset)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
