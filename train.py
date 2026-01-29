import numpy as np
from neural_network import NeuralNetwork
from data_Loader import load_mnist

GREEN = "\033[92m"
RED   = "\033[91m"
# -------------------------------
# Hyperparameters
# -------------------------------
EPOCHS = 20
LEARNING_RATE = 0.008
MODEL_PATH = "mnist_model.npz"

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
# Training
# -------------------------------
print("\nStarting training...\n")

best_accuracy = 0

for epoch in range(EPOCHS):
    correct = 0

    for image, label in train_dataset:
        X = image.numpy().reshape(1, 784)
        y = label

        output = nn.forward(X)
        prediction = np.argmax(output)

        if prediction == y:
            correct += 1

        nn.backward(X, y, LEARNING_RATE)

    accuracy = (correct / len(train_dataset)) * 100
    
    if accuracy < best_accuracy:
        color = RED
        
    else:
        color = GREEN
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Accuracy: {color}{accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        nn.save(MODEL_PATH)
        print("✓ Best model saved")

# -------------------------------
# Testing
# -------------------------------
print("\nLoading best model for testing...\n")
nn.load(MODEL_PATH)

correct = 0
for image, label in test_dataset:
    X = image.numpy().reshape(1, 784)
    output = nn.forward(X)
    prediction = np.argmax(output)

    if prediction == label:
        correct += 1

test_accuracy = (correct / len(test_dataset)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
