import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return dataset


def show_image_by_index(dataset, index):
    image, label = dataset[index]

    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Index: {index} | Label: {label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    dataset = load_mnist()

    index = int(input("Enter MNIST image index: "))
    show_image_by_index(dataset, index)
