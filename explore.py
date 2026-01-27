# Import libraries
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("Starting MNIST exploration...")

# Step 1: Define how to transform images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
])

# Step 2: Download MNIST dataset (this will take a minute first time)
print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")

# Step 3: Look at one image
image, label = train_dataset[0]
print(f"\nFirst image shape: {image.shape}")
print(f"First image label: {label}")

# Step 4: Visualize 10 images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample MNIST Digits', fontsize=16)

for i in range(10):
    ax = axes[i // 5, i % 5]
    image, label = train_dataset[i]
    
    # Display image
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png')
print("\nSaved visualization to 'mnist_samples.png'")
plt.show()

print("\n✅ SUCCESS! You just loaded and visualized MNIST dataset!")