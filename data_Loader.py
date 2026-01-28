# Import libraries
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def load_mnist(visualize=False, num_samples=10):
    """
    Load MNIST dataset and optionally visualize samples.
    
    Args:
        visualize (bool): Whether to visualize sample images
        num_samples (int): Number of samples to visualize if visualize=True
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print("Starting MNIST data loading...")

    # Step 1: Define how to transform images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor (0-1 range)
    ])

    # Step 2: Download MNIST dataset (this will take a minute first time)
    print("Loading MNIST dataset...")
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

    print(f"✓ Training images: {len(train_dataset)}")
    print(f"✓ Test images: {len(test_dataset)}")

    # Step 3: Display dataset info
    image, label = train_dataset[0]
    print(f"\nDataset Info:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Pixel value range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  First image label: {label}")

    # Step 4: Visualize samples if requested
    if visualize:
        visualize_samples(train_dataset, num_samples)

    print("\n✅ SUCCESS! MNIST dataset loaded successfully!")
    
    return train_dataset, test_dataset


def visualize_samples(dataset, num_samples=10, save_path='mnist_samples.png'):
    """
    Visualize sample images from the dataset.
    
    Args:
        dataset: The MNIST dataset
        num_samples (int): Number of samples to display
        save_path (str): Path to save the visualization
    """
    rows = 2
    cols = (num_samples + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle('Sample MNIST Digits', fontsize=16)

    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        image, label = dataset[i]
        
        # Display image
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to '{save_path}'")
    plt.close()


def get_sample_batch(dataset, batch_size=32):
    """
    Get a batch of samples from the dataset.
    
    Args:
        dataset: The MNIST dataset
        batch_size (int): Number of samples in the batch
    
    Returns:
        tuple: (images, labels) as numpy arrays
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))
    
    return images.numpy(), labels.numpy()


# Main execution (only runs when script is executed directly)
if __name__ == "__main__":
    # Load and visualize dataset
    train_dataset, test_dataset = load_mnist(visualize=True, num_samples=10)
    
    # Optional: Get a sample batch
    print("\nGetting a sample batch...")
    images, labels = get_sample_batch(train_dataset, batch_size=5)
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch labels: {labels}")