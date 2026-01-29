import numpy as np
from neural_network import NeuralNetwork
from data_Loader import load_mnist

# ---------- COLORS ----------
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

# ---------- LOAD DATA ----------
_, test_dataset = load_mnist()

# ---------- LOAD TRAINED MODEL ----------
nn = NeuralNetwork()
nn.load("mnist_model.npz")

print("\nStarting detailed test analysis...\n")

correct = 0
total = 0

for idx, (image, label) in enumerate(test_dataset):
    X = image.numpy().reshape(1, 784)

    output = nn.forward(X)
    prediction = np.argmax(output)
    confidence = np.max(output) * 100

    total += 1

    if prediction == label:
        correct += 1
        print(
            f"{GREEN}{BOLD}[✓]{RESET} "
            f"Index: {idx:5d} | "
            f"True: {label} | "
            f"Pred: {prediction} | "
            f"Conf: {confidence:6.2f}%"
        )
    else:
        print(
            f"{RED}{BOLD}[✗]{RESET} "
            f"Index: {idx:5d} | "
            f"True: {label} | "
            f"Pred: {prediction} | "
            f"Conf: {confidence:6.2f}%"
        )

print("\n" + "=" * 60)
print(f"Final Accuracy: {(correct / total) * 100:.2f}%")
print("=" * 60)
