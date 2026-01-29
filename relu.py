import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_Loader import load_mnist

EPOCHS = 20
LEARNING_RATE = 0.008

# -------------------------------
# Loss function (Cross Entropy)
# -------------------------------
def cross_entropy(output, label):
    eps = 1e-9
    return -np.log(output[0, label] + eps)

# -------------------------------
# Load data
# -------------------------------
print("Loading MNIST dataset...")
train_dataset, test_dataset = load_mnist()

# -------------------------------
# Initialize network
# -------------------------------
nn = NeuralNetwork()

train_losses = []
test_losses = []

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(EPOCHS):
    train_loss = 0

    for image, label in train_dataset:
        X = image.numpy().reshape(1, 784)

        output = nn.forward(X)
        train_loss += cross_entropy(output, label)

        nn.backward(X, label, LEARNING_RATE)

    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    # -------------------------------
    # Test loss (NO backward)
    # -------------------------------
    test_loss = 0
    for image, label in test_dataset:
        X = image.numpy().reshape(1, 784)
        output = nn.forward(X)
        test_loss += cross_entropy(output, label)

    test_loss /= len(test_dataset)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

# -------------------------------
# Plot loss curves
# -------------------------------
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Test Loss Curve")
plt.legend()
plt.grid(True)
plt.show()
