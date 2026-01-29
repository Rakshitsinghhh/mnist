import numpy as np
from neural_network import NeuralNetwork
from data_Loader import load_mnist

# -------------------------------
# Hyperparameters
# -------------------------------
EPOCHS = 3
LEARNING_RATE = 0.1   # ↓ lower = more stable
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
# Training loop
# -------------------------------
print("\nStarting training...\n")

best_accuracy = 0

for epoch in range(EPOCHS):
    correct = 0

    for image, label in train_dataset:
        X = image.numpy().reshape(1, 784)
        y = label

        # Forward
        output = nn.forward(X)
        prediction = np.argmax(output)

        if prediction == y:
            correct += 1

        # Learn
        nn.backward(X, y, lr=LEARNING_RATE)

    accuracy = (correct / len(train_dataset)) * 100
    print(f"Epoch {epoch + 1}/{EPOCHS} - Training Accuracy: {accuracy:.2f}%")

    # ✅ Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        nn.save(MODEL_PATH)
        print("✓ Best model saved")

# -------------------------------
# Load best model before testing
# -------------------------------
print("\nLoading best trained model for testing...\n")
nn.load(MODEL_PATH)

# -------------------------------
# Testing loop
# -------------------------------
correct = 0

for image, label in test_dataset:
    X = image.numpy().reshape(1, 784)
    output = nn.forward(X)
    prediction = np.argmax(output)

    if prediction == label:
        correct += 1

test_accuracy = (correct / len(test_dataset)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
