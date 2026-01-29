import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_Loader import load_mnist

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

# -------------------------------
# Hyperparameters
# -------------------------------
EPOCHS = 20
LEARNING_RATE = 0.008
MODEL_PATH = "mnist_model.npz"

# -------------------------------
# Loss function
# -------------------------------
def cross_entropy(output, label):
    eps = 1e-9
    return -np.log(output[0, label] + eps)

# -------------------------------
# Progress bar
# -------------------------------
def progress_bar(epoch, total):
    percent = int((epoch / total) * 100)
    filled = percent // 5
    bar = "█" * filled + "░" * (20 - filled)
    print(f"\rEpoch Progress [{bar}] {percent}%", end="")

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

train_accuracies = []
train_losses = []

print("\nStarting training...\n")
best_accuracy = 0

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1, EPOCHS + 1):
    correct = 0
    total_loss = 0

    for image, label in train_dataset:
        X = image.numpy().reshape(1, 784)
        y = label

        output = nn.forward(X)
        prediction = np.argmax(output)

        if prediction == y:
            correct += 1

        total_loss += cross_entropy(output, y)
        nn.backward(X, y, LEARNING_RATE)

    accuracy = (correct / len(train_dataset)) * 100
    avg_loss = total_loss / len(train_dataset)

    train_accuracies.append(accuracy)
    train_losses.append(avg_loss)

    progress_bar(epoch, EPOCHS)

    color = GREEN if accuracy >= best_accuracy else RED
    print(f"\nEpoch {epoch}/{EPOCHS} - Accuracy: {color}{accuracy:.2f}%{RESET} | Loss: {avg_loss:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        nn.save(MODEL_PATH)
        print("✓ Best model saved")

print("\nTraining completed ✔\n")

# -------------------------------
# Testing
# -------------------------------
print("Loading best model for testing...\n")
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

# -------------------------------
# Graphs
# -------------------------------
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracies, marker='o')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
