"""
Phase 2: Building Your First Neural Network

This script helps you understand:
- How to define a neural network architecture
- What layers, neurons, weights, and biases are
- How to make predictions (forward pass)
- Network structure and parameter count

This network won't be trained yet - that comes in Phase 3!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("=" * 60)
print("Building Your First Neural Network")
print("=" * 60)

# Step 1: Define the Neural Network Architecture
print("\n[1] Defining the network architecture...")


class SimpleNN(nn.Module):
    """
    A simple fully-connected (dense) neural network.

    Architecture:
        Input Layer:  784 neurons (28×28 pixels)
                ↓
        Hidden Layer: 128 neurons + ReLU activation
                ↓
        Output Layer: 10 neurons (digits 0-9) + Softmax

    Each neuron in a layer is connected to every neuron in the next layer.
    """

    def __init__(self):
        super(SimpleNN, self).__init__()

        # Layer 1: Input (784) → Hidden (128)
        self.fc1 = nn.Linear(784, 128)
        # This creates:
        #   - Weight matrix: 784 × 128 = 100,352 parameters
        #   - Bias vector: 128 parameters
        #   Total: 100,480 parameters

        # Layer 2: Hidden (128) → Output (10)
        self.fc2 = nn.Linear(128, 10)
        # This creates:
        #   - Weight matrix: 128 × 10 = 1,280 parameters
        #   - Bias vector: 10 parameters
        #   Total: 1,290 parameters

    def forward(self, x):
        """
        Forward pass: how data flows through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            output: Predictions of shape (batch_size, 10)
        """
        # Flatten the image from (batch_size, 1, 28, 28) to (batch_size, 784)
        x = x.view(-1, 784)
        # -1 means "figure out this dimension" (it will be batch_size)
        # 784 = 28 × 28 (all pixels in one row)

        # Pass through first layer and apply ReLU activation
        x = self.fc1(x)  # Shape: (batch_size, 128)
        x = F.relu(x)    # ReLU: max(0, x) - introduces non-linearity
        # ReLU zeros out negative values, keeps positive values

        # Pass through second layer
        x = self.fc2(x)  # Shape: (batch_size, 10)

        # Apply softmax to get probabilities
        # Softmax converts raw scores to probabilities that sum to 1
        output = F.softmax(x, dim=1)
        # dim=1 means apply softmax across the 10 digit classes

        return output


# Create an instance of the network
model = SimpleNN()

print("    ✓ Network architecture defined!")
print("\n    Network structure:")
print("    " + "─" * 50)
print(f"    {model}")
print("    " + "─" * 50)

# Step 2: Count the parameters
print("\n[2] Counting network parameters...")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n    Total parameters: {total_params:,}")
print(f"    Trainable parameters: {trainable_params:,}")
print("\n    Breakdown by layer:")

for name, param in model.named_parameters():
    print(f"      • {name:15s}: {param.numel():>7,} parameters, shape {tuple(param.shape)}")

print("\n    💡 These parameters (weights and biases) are what the")
print("       network will learn during training!")

# Step 3: Visualize the network architecture
print("\n[3] Visualizing network architecture...")

fig, ax = plt.subplots(figsize=(12, 8))

# Layer positions
layers = ['Input\n(784)', 'Hidden\n(128)', 'Output\n(10)']
layer_sizes = [784, 128, 10]
x_positions = [0, 1, 2]

# Draw layers as rectangles
for x, size, label in zip(x_positions, layer_sizes, layers):
    # Rectangle height proportional to layer size (with scaling)
    height = min(size / 100, 8)  # Cap height for visualization
    rect = plt.Rectangle((x - 0.1, -height/2), 0.2, height,
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Label
    ax.text(x, -height/2 - 0.8, label, ha='center', va='top',
            fontsize=12, fontweight='bold')

    # Show number of neurons
    ax.text(x, height/2 + 0.3, f'{size} neurons', ha='center', va='bottom',
            fontsize=10, style='italic', color='darkblue')

# Draw connections
for i in range(len(x_positions) - 1):
    x1, x2 = x_positions[i], x_positions[i + 1]
    ax.plot([x1 + 0.1, x2 - 0.1], [0, 0], 'k-', alpha=0.3, linewidth=1)
    ax.plot([x1 + 0.1, x2 - 0.1], [1, 0.5], 'k-', alpha=0.1, linewidth=0.5)
    ax.plot([x1 + 0.1, x2 - 0.1], [-1, -0.5], 'k-', alpha=0.1, linewidth=0.5)

    # Label connections
    mid_x = (x1 + x2) / 2
    if i == 0:
        ax.text(mid_x, 1.8, 'fc1: 784→128\n(100,480 params)',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(mid_x, -2.5, 'ReLU\nActivation',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax.text(mid_x, 1.8, 'fc2: 128→10\n(1,290 params)',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(mid_x, -2.5, 'Softmax\nActivation',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-4, 4)
ax.axis('off')
ax.set_title('Simple Neural Network Architecture\nTotal: 101,770 Parameters',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('MNIST/visualizations/network_architecture.png', dpi=150, bbox_inches='tight')
print("    ✓ Saved: MNIST/visualizations/network_architecture.png")

# Step 4: Load a sample image and make a prediction
print("\n[4] Making a prediction (before training)...")

# Load MNIST dataset
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

# Get a random image
idx = torch.randint(0, len(test_dataset), (1,)).item()
image, true_label = test_dataset[idx]

print(f"\n    • Selected random test image #{idx}")
print(f"    • True label: {true_label}")

# Make prediction (no training yet, so it will be random!)
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Don't compute gradients for inference
    # Add batch dimension: (1, 28, 28) → (1, 1, 28, 28)
    image_batch = image.unsqueeze(0)
    predictions = model(image_batch)

# Get predicted probabilities
probs = predictions[0].numpy()  # Remove batch dimension
predicted_digit = probs.argmax()

print(f"\n    Prediction probabilities for each digit:")
for digit in range(10):
    prob = probs[digit] * 100
    bar = "█" * int(prob / 2)  # Scale for display
    indicator = " ← PREDICTED" if digit == predicted_digit else ""
    correct = " ✓ CORRECT!" if digit == true_label else ""
    print(f"      {digit}: {prob:5.1f}% {bar}{indicator}{correct}")

print(f"\n    • Predicted digit: {predicted_digit}")
print(f"    • Confidence: {probs[predicted_digit] * 100:.1f}%")

if predicted_digit == true_label:
    print("    • Status: CORRECT (lucky guess!)")
else:
    print(f"    • Status: WRONG (expected {true_label})")

print("\n    💡 The network hasn't been trained yet, so predictions")
print("       are essentially random! Training will fix this.")

# Visualize the prediction
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Untrained Network Prediction (Random Weights)',
             fontsize=14, fontweight='bold')

# Show the image
axes[0].imshow(image.squeeze(), cmap='gray')
axes[0].set_title(f'Input Image (True Label: {true_label})', fontsize=12)
axes[0].axis('off')

# Show prediction probabilities
colors = ['green' if i == true_label else 'steelblue' for i in range(10)]
colors[predicted_digit] = 'red' if predicted_digit != true_label else 'darkgreen'

bars = axes[1].barh(range(10), probs * 100, color=colors, alpha=0.7)
axes[1].set_xlabel('Probability (%)', fontsize=12)
axes[1].set_ylabel('Digit', fontsize=12)
axes[1].set_title('Prediction Probabilities (Untrained Network)', fontsize=12)
axes[1].set_yticks(range(10))
axes[1].set_xlim(0, 100)
axes[1].grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='True Label'),
    Patch(facecolor='red', alpha=0.7, label='Wrong Prediction'),
    Patch(facecolor='steelblue', alpha=0.7, label='Other Digits')
]
axes[1].legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('MNIST/visualizations/untrained_prediction.png', dpi=150, bbox_inches='tight')
print("\n    ✓ Saved: MNIST/visualizations/untrained_prediction.png")

# Step 5: Examine the weights
print("\n[5] Looking at initial weights...")

# Get weights from first layer
first_layer_weights = model.fc1.weight.data  # Shape: (128, 784)

print(f"    • First layer weight matrix shape: {tuple(first_layer_weights.shape)}")
print(f"    • This means: 128 neurons, each looking at 784 pixels")
print(f"\n    • Weight statistics:")
print(f"      - Mean: {first_layer_weights.mean().item():.6f}")
print(f"      - Std Dev: {first_layer_weights.std().item():.6f}")
print(f"      - Min: {first_layer_weights.min().item():.6f}")
print(f"      - Max: {first_layer_weights.max().item():.6f}")

# Visualize some weight patterns
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Initial Weights of 10 Hidden Neurons (Before Training)',
             fontsize=14, fontweight='bold')

for idx in range(10):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]

    # Get weights for this neuron and reshape to 28×28
    neuron_weights = first_layer_weights[idx].reshape(28, 28).numpy()

    im = ax.imshow(neuron_weights, cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax.set_title(f'Neuron {idx}', fontsize=10)
    ax.axis('off')

plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('MNIST/visualizations/initial_weights.png', dpi=150, bbox_inches='tight')
print("\n    ✓ Saved: MNIST/visualizations/initial_weights.png")

print("\n    💡 These weights are randomly initialized. They don't")
print("       detect any meaningful patterns yet. After training,")
print("       they'll learn to detect edges, curves, and features!")

# Summary
print("\n" + "=" * 60)
print("NETWORK BUILDING COMPLETE!")
print("=" * 60)
print("\nKey Takeaways:")
print("  1. Our network has 3 layers: Input (784) → Hidden (128) → Output (10)")
print(f"  2. Total of {total_params:,} parameters to learn")
print("  3. ReLU activation adds non-linearity (crucial for learning)")
print("  4. Softmax converts outputs to probabilities")
print("  5. Untrained network makes random predictions (~10% accuracy)")
print("\nNext Step:")
print("  → Run: python 03_train_model.py")
print("  → This will train the network and watch it learn!")
print("=" * 60)

plt.show()
