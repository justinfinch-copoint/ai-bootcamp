"""
Phase 5: Convolutional Neural Networks (CNNs)

This script helps you understand:
- Why CNNs work better for images than fully-connected networks
- Convolutional layers and feature maps
- Pooling layers
- Achieving state-of-the-art results (>99% accuracy)
- Visualizing what the network learns

This is the modern approach to image classification!
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

print("=" * 60)
print("Convolutional Neural Network (CNN)")
print("=" * 60)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[1] Using device: {device}")

# Step 1: Define CNN Architecture
print("\n[2] Defining CNN architecture...")


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST

    Architecture:
        Input: 28×28 grayscale image

        Conv Block 1:
            - Conv2d: 1 → 32 channels, 3×3 kernel
            - ReLU activation
            - MaxPool: 2×2 (reduces to 14×14)

        Conv Block 2:
            - Conv2d: 32 → 64 channels, 3×3 kernel
            - ReLU activation
            - MaxPool: 2×2 (reduces to 7×7)

        Fully Connected:
            - Flatten: 7×7×64 = 3,136 features
            - FC1: 3,136 → 128
            - Dropout: 50% (regularization)
            - FC2: 128 → 10 (output classes)
    """

    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale input
            out_channels=32,    # 32 different filters/feature maps
            kernel_size=3,      # 3×3 filter
            padding=1           # Keep same size after convolution
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,     # Input from previous layer
            out_channels=64,    # 64 different filters
            kernel_size=3,
            padding=1
        )

        # Pooling layer (will be reused)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2 pooling

        # Fully connected layers
        # After two poolings: 28→14→7, with 64 channels: 7×7×64 = 3,136
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        # Randomly drop 50% of neurons during training
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv Block 1: Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))  # 28×28×1 → 14×14×32

        # Conv Block 2: Conv → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # 14×14×32 → 7×7×64

        # Flatten for fully connected layers
        x = x.view(-1, 7 * 7 * 64)  # Flatten to (batch_size, 3136)

        # Fully connected layers
        x = F.relu(self.fc1(x))     # → 128
        x = self.dropout(x)          # Apply dropout
        x = self.fc2(x)              # → 10

        return x


model = CNN().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"    ✓ CNN created with {total_params:,} parameters")
print("\n    Network structure:")
print("    " + "─" * 50)
print(f"    {model}")
print("    " + "─" * 50)

# Compare with simple network
simple_nn_params = 101_770  # From Phase 2
print(f"\n    💡 Comparison:")
print(f"       Simple NN: {simple_nn_params:,} parameters")
print(f"       CNN: {total_params:,} parameters")
print(
    f"       Difference: {total_params - simple_nn_params:,} more parameters")
print(f"       But CNNs are more efficient for images!")

# Step 2: Prepare data with augmentation
print("\n[3] Preparing data with augmentation...")

# Training transforms: add augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate by up to ±10 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test transforms: just normalize
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=test_transform)

# Split training data
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size])

# Data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"    ✓ Data loaded with augmentation")
print(f"    💡 Data augmentation helps prevent overfitting by creating")
print(f"       slight variations of training images (rotations, etc.)")

# Step 3: Setup training
print("\n[4] Setting up training...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler: reduce LR when progress plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

print("    ✓ Using Adam optimizer with learning rate scheduling")

# Step 4: Training loop
print("\n[5] Training CNN...")

num_epochs = 12
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print(f"    • Epochs: {num_epochs}")
print("\n" + "─" * 60)


def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]')

    for batch_idx, (images, labels) in enumerate(pbar, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/batch_idx:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]  ')

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar, 1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/batch_idx:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return running_loss / len(val_loader), 100. * correct / total


# Training loop
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(epoch)
    val_loss, val_acc = validate(epoch)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    print(f"    Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print("─" * 60)

training_time = time.time() - start_time
print(f"\n✓ Training complete! Time: {training_time:.1f}s")

# Step 5: Evaluate on test set
print("\n[6] Evaluating on test set...")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100. * correct / total
print(f"\n    CNN Test Accuracy: {test_accuracy:.2f}%")
print(f"    (Compare with ~97-98% from simple network!)")

# Step 6: Visualize training
print("\n[7] Visualizing training progress...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN Training Progress', fontsize=14, fontweight='bold')

epochs = range(1, num_epochs + 1)

ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_accuracies, 'b-',
         label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_accuracies, 'r-',
         label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([95, 100])

plt.tight_layout()
plt.savefig('visualizations/cnn_training_progress.png',
            dpi=150, bbox_inches='tight')
print("    ✓ Saved: visualizations/cnn_training_progress.png")

# Step 7: Visualize learned filters
print("\n[8] Visualizing learned convolutional filters...")

# Get first layer filters
first_layer_filters = model.conv1.weight.data.cpu()  # Shape: (32, 1, 3, 3)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle('First Layer Convolutional Filters (32 filters, 3×3 each)',
             fontsize=14, fontweight='bold')

for idx in range(32):
    row = idx // 8
    col = idx % 8
    ax = axes[row, col]

    # Get filter and normalize for visualization
    filt = first_layer_filters[idx, 0].numpy()

    im = ax.imshow(filt, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title(f'F{idx}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations/cnn_filters.png', dpi=150, bbox_inches='tight')
print("    ✓ Saved: visualizations/cnn_filters.png")
print("    💡 These filters learn to detect edges, corners, and curves!")

# Step 8: Visualize feature maps
print("\n[9] Visualizing feature maps...")

# Get a sample image
sample_img, sample_label = test_dataset[0]
sample_img_batch = sample_img.unsqueeze(0).to(device)

# Get activations from first conv layer
model.eval()
with torch.no_grad():
    # Hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations['conv1'] = output

    handle = model.conv1.register_forward_hook(hook_fn)
    _ = model(sample_img_batch)
    handle.remove()

# Get feature maps
feature_maps = activations['conv1'].squeeze(0).cpu()  # Shape: (32, 28, 28)

# Visualize
fig, axes = plt.subplots(5, 7, figsize=(14, 10))
fig.suptitle(f'Feature Maps from First Conv Layer (Digit: {sample_label})',
             fontsize=14, fontweight='bold')

# Show original image first
axes[0, 0].imshow(sample_img.squeeze(), cmap='gray')
axes[0, 0].set_title('Original', fontsize=10)
axes[0, 0].axis('off')

# Show first 32 feature maps (skip first cell)
for idx in range(32):
    row = (idx + 1) // 7
    col = (idx + 1) % 7
    ax = axes[row, col]

    fmap = feature_maps[idx].numpy()
    ax.imshow(fmap, cmap='viridis')
    ax.set_title(f'FM{idx}', fontsize=8)
    ax.axis('off')

# Hide extra subplots
for idx in range(33, 35):
    row = idx // 7
    col = idx % 7
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('visualizations/cnn_feature_maps.png',
            dpi=150, bbox_inches='tight')
print("    ✓ Saved: visualizations/cnn_feature_maps.png")
print("    💡 Different feature maps detect different patterns!")

# Step 9: Save model
print("\n[10] Saving model...")

os.makedirs('MNIST/models', exist_ok=True)

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_accuracy,
}, 'models/cnn_trained.pth')

print("    ✓ Saved: MNIST/models/cnn_trained.pth")

# Step 10: Show predictions
print("\n[11] Sample predictions...")

# Get batch
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    _, predictions = outputs.max(1)

# Move to CPU
images = images.cpu()
labels = labels.cpu()
predictions = predictions.cpu()
probabilities = probabilities.cpu()

# Plot
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle(f'CNN Predictions (Test Accuracy: {test_accuracy:.2f}%)',
             fontsize=14, fontweight='bold')

for idx in range(10):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]

    image = images[idx].squeeze()
    true_label = labels[idx].item()
    pred_label = predictions[idx].item()
    confidence = probabilities[idx][pred_label].item() * 100

    ax.imshow(image, cmap='gray')

    color = 'green' if pred_label == true_label else 'red'
    status = '✓' if pred_label == true_label else '✗'

    ax.set_title(f'{status} True: {true_label}, Pred: {pred_label}\n({confidence:.1f}%)',
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations/cnn_predictions.png', dpi=150, bbox_inches='tight')
print("    ✓ Saved: MNIST/visualizations/cnn_predictions.png")

# Summary
print("\n" + "=" * 60)
print("CNN TRAINING COMPLETE!")
print("=" * 60)
print("\nResults Summary:")
print(f"  • Final validation accuracy: {val_accuracies[-1]:.2f}%")
print(f"  • Test accuracy: {test_accuracy:.2f}%")
print(f"  • Improvement over simple NN: ~{test_accuracy - 97.5:.1f}%")
print(f"  • Total parameters: {total_params:,}")
print(f"  • Training time: {training_time:.1f}s")
print("\nWhy CNNs Work Better:")
print("  1. Spatial structure: Understands that nearby pixels are related")
print("  2. Translation invariance: Recognizes patterns anywhere in image")
print("  3. Hierarchical features: Simple edges → complex shapes")
print("  4. Parameter efficiency: Shared weights across image")
print("  5. Regularization: Dropout prevents overfitting")
print("\nWhat You Learned:")
print("  • Convolutional layers detect local patterns")
print("  • Pooling reduces size while keeping important features")
print("  • CNNs learn hierarchical representations")
print("  • Data augmentation helps generalization")
print("  • Learning rate scheduling improves training")
print("\nNext Steps:")
print("  → Experiment with the architecture:")
print("    - Add more conv layers")
print("    - Try different filter sizes (5×5, 7×7)")
print("    - Adjust dropout rate")
print("  → Try other datasets (Fashion-MNIST, CIFAR-10)")
print("  → Learn about modern architectures (ResNet, VGG, etc.)")
print("=" * 60)

plt.show()
