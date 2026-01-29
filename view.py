import csv
import matplotlib.pyplot as plt
from data_Loader import load_mnist

# Load dataset
train_dataset, test_dataset = load_mnist()
dataset = test_dataset   # change to train_dataset if needed

CSV_FILE = "mnist_preview.csv"
NUM_IMAGES = 100   # how many images to save & preview

# -------------------------------
# Save to CSV
# -------------------------------
print("Saving MNIST data to CSV...")

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    # Header
    header = ["index", "label"] + [f"pixel_{i}" for i in range(784)]
    writer.writerow(header)

    for idx in range(NUM_IMAGES):
        image, label = dataset[idx]
        pixels = image.numpy().reshape(784) * 255  # back to 0–255

        row = [idx, label]
        writer.writerow(row)

print(f"Saved {NUM_IMAGES} images to {CSV_FILE}")

# -------------------------------
# Visualize images
# -------------------------------
print("Displaying images...")

plt.figure(figsize=(10, 4))
for i in range(min(10, NUM_IMAGES)):
    image, label = dataset[i]

    plt.subplot(1, 10, i + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Idx:{i}\nLbl:{label}")
    plt.axis("off")

plt.show()
