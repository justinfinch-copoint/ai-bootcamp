"""
Phase 1: Exploring the MNIST Dataset

This script helps you understand:
- How to load the MNIST dataset
- What the data looks like (shape, type)
- How images are represented as numbers
- Visualizing sample digits

Run this first to get familiar with what you're working with!
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 60)
print("MNIST Dataset Exploration")
print("=" * 60)

# Step 1: Download and load the MNIST dataset
print("\n[1] Loading MNIST dataset...")
print("    (This will download ~10MB on first run)")

# Transform to convert images to PyTorch tensors
transform = transforms.ToTensor()

# Download training and test datasets
train_dataset = datasets.MNIST(
    root='./data',           # Where to save the data
    train=True,              # Get training data
    download=True,           # Download if not present
    transform=transform      # Convert to tensor
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,             # Get test data
    download=True,
    transform=transform
)

print(f"    ✓ Training samples: {len(train_dataset):,}")
print(f"    ✓ Test samples: {len(test_dataset):,}")

# Step 2: Understand the data structure
print("\n[2] Understanding the data structure...")

# Get one sample
image, label = train_dataset[0]

print(f"    • Each image shape: {image.shape}")
print(f"      - Channels: {image.shape[0]} (grayscale)")
print(f"      - Height: {image.shape[1]} pixels")
print(f"      - Width: {image.shape[2]} pixels")
print(f"    • Image data type: {image.dtype}")
print(f"    • Value range: [{image.min():.2f}, {image.max():.2f}]")
print(f"    • Label: {label} (the actual digit)")

# Step 3: Visualize some examples
print("\n[3] Visualizing sample digits...")

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Create a figure with 10 samples (one of each digit)
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Sample MNIST Digits (One of Each Class)', fontsize=14, fontweight='bold')

# Find one example of each digit (0-9)
examples = {}
for image, label in train_dataset:
    if label not in examples:
        examples[label] = image
    if len(examples) == 10:
        break

# Plot each digit
for idx, (digit, image) in enumerate(sorted(examples.items())):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]

    # Remove channel dimension for plotting (1, 28, 28) -> (28, 28)
    img_to_plot = image.squeeze()

    ax.imshow(img_to_plot, cmap='gray')
    ax.set_title(f'Digit: {digit}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations/sample_digits.png', dpi=150, bbox_inches='tight')
print("    ✓ Saved: visualizations/sample_digits.png")

# Step 4: Look at the actual pixel values
print("\n[4] Looking at pixel values for one image...")

# Get a single image
sample_image, sample_label = train_dataset[0]
sample_pixels = sample_image.squeeze().numpy()

print(f"    • This is a '{sample_label}'")
print(f"    • Image as a 28×28 grid of numbers:")
print(f"    • (showing center 10×10 region for readability)\n")

# Show a 10x10 center region
center_region = sample_pixels[9:19, 9:19]
for row in center_region:
    print("    ", " ".join(f"{val:.1f}" for val in row))

print(f"\n    • Values close to 0.0 = black (background)")
print(f"    • Values close to 1.0 = white (the digit)")

# Step 5: Visualize one image in detail
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Detailed Look at a Single Image (Label: {sample_label})',
             fontsize=14, fontweight='bold')

# Original image
axes[0].imshow(sample_pixels, cmap='gray')
axes[0].set_title('Original Image (28×28)', fontsize=12)
axes[0].axis('off')

# With pixel grid
axes[1].imshow(sample_pixels, cmap='gray', interpolation='nearest')
axes[1].set_title('With Pixel Boundaries', fontsize=12)
axes[1].grid(True, which='both', color='red', linewidth=0.5, alpha=0.3)
axes[1].set_xticks(np.arange(-0.5, 28, 1))
axes[1].set_yticks(np.arange(-0.5, 28, 1))
axes[1].tick_params(labelbottom=False, labelleft=False)

# As heatmap with values
# Show only 14×14 center for readability
center_14 = sample_pixels[7:21, 7:21]
im = axes[2].imshow(center_14, cmap='hot', interpolation='nearest')
axes[2].set_title('Pixel Values (14×14 Center)', fontsize=12)
plt.colorbar(im, ax=axes[2], fraction=0.046)
axes[2].set_xticks(range(14))
axes[2].set_yticks(range(14))
axes[2].tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('visualizations/detailed_image.png', dpi=150, bbox_inches='tight')
print("\n    ✓ Saved: visualizations/detailed_image.png")

# Step 6: Class distribution
print("\n[5] Analyzing class distribution...")

# Count examples of each digit
digit_counts = torch.zeros(10, dtype=torch.int)
for _, label in train_dataset:
    digit_counts[label] += 1

print("\n    Distribution of digits in training set:")
for digit in range(10):
    count = digit_counts[digit].item()
    percentage = (count / len(train_dataset)) * 100
    bar = "█" * int(percentage)
    print(f"    Digit {digit}: {count:,} ({percentage:.1f}%) {bar}")

# Visualize distribution
fig, ax = plt.subplots(figsize=(10, 6))
digits = range(10)
ax.bar(digits, digit_counts.numpy(), color='steelblue', alpha=0.8)
ax.set_xlabel('Digit', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Distribution of Digits in MNIST Training Set', fontsize=14, fontweight='bold')
ax.set_xticks(digits)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, count in enumerate(digit_counts):
    ax.text(i, count + 100, str(count.item()), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('visualizations/class_distribution.png', dpi=150, bbox_inches='tight')
print("\n    ✓ Saved: visualizations/class_distribution.png")

# Step 7: Show variety within one class
print("\n[6] Showing variety within one digit class...")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('10 Different Handwritten "7"s - See the Variation!',
             fontsize=14, fontweight='bold')

# Find 10 examples of the digit "7"
sevens = []
for image, label in train_dataset:
    if label == 7:
        sevens.append(image)
    if len(sevens) == 10:
        break

# Plot them
for idx, image in enumerate(sevens):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f'Example {idx + 1}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations/digit_variety.png', dpi=150, bbox_inches='tight')
print("    ✓ Saved: visualizations/digit_variety.png")

# Summary
print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
print("\nKey Takeaways:")
print("  1. Images are just 28×28 grids of numbers (0.0 to 1.0)")
print("  2. Each image has a label (0-9) - this is supervised learning")
print("  3. ~60,000 training examples, ~10,000 test examples")
print("  4. Classes are roughly balanced (~10% each)")
print("  5. Significant variation even within one digit class")
print("\nNext Step:")
print("  → Check the saved visualizations in visualizations/")
print("  → Then run: python 02_simple_nn.py")
print("=" * 60)

# Make sure to show plots (if running interactively)
plt.show()
