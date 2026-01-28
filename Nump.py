import numpy as np
from neural_network import NeuralNetwork
from data_Loader import load_mnist  # Updated import name (with underscore)




# Step 1: Load MNIST dataset
print("Loading MNIST dataset...")
train_dataset, test_dataset = load_mnist()
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Step 2: Initialize neural network
print("\nInitializing neural network...")
nn = NeuralNetwork()


# Step 3: Pick a few MNIST samples for testing (let's take 5)
num_samples = 5
print(f"\nTesting with {num_samples} samples from training set:")

for i in range(num_samples):
    image, label = train_dataset[i]
    
    # Flatten image to 1D array (28x28 -> 784)
    X = image.numpy().reshape(1, 784)
    y = label
    
    # Step 4: Forward pass
    output = nn.forward(X)
    predicted_class = np.argmax(output)
    
    print(f"\n{'='*50}")
    print(f"Sample {i+1}")
    print(f"{'='*50}")
    print(f"True label: {y}")
    print(f"Predicted class: {predicted_class}")
    print(f"Forward pass output (first 10 elements): {output.flatten()[:10]}")
    print(f"Output shape: {output.shape}")
    
    count =0
    if y==predicted_class:
        count+=1
    
    # Step 5: Backward pass
    nn.backward(X, y, lr=0.01)
    print("✓ Backward pass executed successfully!")
    
print("the counter is" , count)

print(f"\n{'='*50}")
print("All tests completed successfully!")
print(f"{'='*50}")