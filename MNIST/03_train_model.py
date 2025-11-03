"""
Phase 3: Training Your Neural Network

This script helps you understand:
- The training process (forward pass, loss calculation, backpropagation)
- Loss functions and optimization
- Training vs validation accuracy
- Hyperparameters (learning rate, batch size, epochs)
- Watching the network learn in real-time!

This is where the magic happens - watch your network go from random
guessing to 97-98% accuracy!
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

print("=" * 60)
print("Training Your Neural Network")
print("=" * 60)

# Step 1: Set up the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[1] Using device: {device}")

if device.type == 'cuda':
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
else:
    print("    (GPU not available, using CPU - training will be slower)")

# Step 2: Define the network (same as before)
print("\n[2] Defining the network...")


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Note: We don't apply softmax here because
        # CrossEntropyLoss does it internally (more numerically stable)
        return x


model = SimpleNN().to(device)
print(
    f"    ✓ Network created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Step 3: Prepare the data
print("\n[3] Preparing datasets...")

# Transform: Convert to tensor and normalize
# Normalization: mean=0.1307, std=0.3081 (computed from MNIST training set)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

# Split training data into train and validation
train_size = int(0.9 * len(train_dataset))  # 90% for training
val_size = len(train_dataset) - train_size  # 10% for validation

train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size])

print(f"    • Training samples: {len(train_dataset):,}")
print(f"    • Validation samples: {len(val_dataset):,}")
print(f"    • Test samples: {len(test_dataset):,}")

# Create data loaders
batch_size = 64  # Process 64 images at a time

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"    • Batch size: {batch_size}")
print(f"    • Batches per epoch: {len(train_loader)}")

print("\n    💡 Why batches? Processing in batches is more efficient than")
print("       one image at a time, and gives better gradient estimates")
print("       than using the entire dataset at once.")

# Step 4: Define loss function and optimizer
print("\n[4] Setting up training components...")

# Loss function: Cross-Entropy Loss
# Measures how far off predictions are from true labels
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (Adaptive Moment Estimation)
# Automatically adjusts learning rate for each parameter
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"    • Loss function: Cross-Entropy Loss")
print(f"    • Optimizer: Adam")
print(f"    • Learning rate: {learning_rate}")

print("\n    💡 Adam is a good default optimizer - it adapts the learning")
print("       rate for each parameter, making training more stable.")

# Step 5: Training loop
print("\n[5] Starting training...")

num_epochs = 10  # How many times to go through the entire dataset

# Track metrics for visualization
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print(f"    • Number of epochs: {num_epochs}")
print("\n" + "─" * 60)


def train_one_epoch(epoch):
    """Train for one epoch and return average loss and accuracy."""
    model.train()  # Set to training mode (enables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]')

    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients (important!)
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(epoch):
    """Evaluate on validation set and return average loss and accuracy."""
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]  ')

    with torch.no_grad():  # Don't compute gradients for validation
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# Training loop
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validate
    val_loss, val_acc = validate(epoch)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Print epoch summary
    print(f"    Epoch {epoch:2d} Summary: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print("─" * 60)

training_time = time.time() - start_time
print(f"\n✓ Training complete! Total time: {training_time:.1f} seconds")

# Step 6: Visualize training progress
print("\n[6] Visualizing training progress...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Progress', fontsize=14, fontweight='bold')

epochs = range(1, num_epochs + 1)

# Plot losses
ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Over Time', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot accuracies
ax2.plot(epochs, train_accuracies, 'b-',
         label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_accuracies, 'r-',
         label='Validation Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Accuracy Over Time', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('visualizations/training_progress.png',
            dpi=150, bbox_inches='tight')
print("    ✓ Saved: MNIST/visualizations/training_progress.png")

# Step 7: Final evaluation on test set
print("\n[7] Final evaluation on test set...")

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
print(f"\n    Test Accuracy: {test_accuracy:.2f}%")

# Step 8: Save the trained model
print("\n[8] Saving the trained model...")

os.makedirs('models', exist_ok=True)

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_accuracy,
}, 'models/simple_nn_trained.pth')

print("    ✓ Saved: MNIST/models/simple_nn_trained.pth")

# Step 9: Show some predictions
print("\n[9] Showing sample predictions...")

# Get a batch from test set
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    _, predictions = outputs.max(1)

# Move back to CPU for plotting
images = images.cpu()
labels = labels.cpu()
predictions = predictions.cpu()
probabilities = probabilities.cpu()

# Plot first 10 predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Sample Predictions from Trained Model',
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

    if pred_label == true_label:
        color = 'green'
        status = '✓'
    else:
        color = 'red'
        status = '✗'

    ax.set_title(f'{status} True: {true_label}, Pred: {pred_label}\n({confidence:.1f}%)',
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations/sample_predictions.png',
            dpi=150, bbox_inches='tight')
print("    ✓ Saved: MNIST/visualizations/sample_predictions.png")

# Step 10: Analyze mistakes
print("\n[10] Analyzing common mistakes...")

# Find all mistakes in test set
model.eval()
mistakes = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)

        # Find incorrect predictions
        incorrect_mask = predictions.ne(labels)
        for idx in torch.where(incorrect_mask)[0]:
            mistakes.append({
                'image': images[idx].cpu(),
                'true': labels[idx].item(),
                'pred': predictions[idx].item(),
                'confidence': F.softmax(outputs[idx], dim=0)[predictions[idx]].item()
            })

print(f"\n    Total mistakes: {len(mistakes)} out of {len(test_dataset)}")
print(f"    Error rate: {len(mistakes)/len(test_dataset)*100:.2f}%")

# Show some mistakes
if len(mistakes) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('Sample Mistakes - Can You See Why?',
                 fontsize=14, fontweight='bold')

    for idx in range(min(10, len(mistakes))):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        mistake = mistakes[idx]
        ax.imshow(mistake['image'].squeeze(), cmap='gray')
        ax.set_title(f'True: {mistake["true"]}, Pred: {mistake["pred"]}\n'
                     f'Confidence: {mistake["confidence"]*100:.1f}%',
                     fontsize=10, color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('visualizations/mistakes.png', dpi=150, bbox_inches='tight')
    print("    ✓ Saved: MNIST/visualizations/mistakes.png")

# Confusion analysis
confusion_pairs = {}
for m in mistakes:
    pair = (m['true'], m['pred'])
    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

print("\n    Most common confusions:")
for (true, pred), count in sorted(confusion_pairs.items(),
                                  key=lambda x: x[1], reverse=True)[:5]:
    print(f"      {true} misclassified as {pred}: {count} times")

# Summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nKey Observations:")
print(f"  1. Final validation accuracy: {val_accuracies[-1]:.2f}%")
print(f"  2. Final test accuracy: {test_accuracy:.2f}%")
print(f"  3. The network learned to recognize digits with ~97-98% accuracy!")
print(f"  4. Training time: {training_time:.1f} seconds")
print(f"  5. Total mistakes: {len(mistakes)} out of {len(test_dataset)}")
print("\nWhat You Learned:")
print("  • How neural networks learn through backpropagation")
print("  • The importance of loss functions and optimizers")
print("  • Why we split data into train/validation/test sets")
print("  • How to track and visualize training progress")
print("\nNext Steps:")
print("  → Experiment with hyperparameters:")
print("    - Try different learning rates (0.01, 0.001, 0.0001)")
print("    - Try different batch sizes (32, 128, 256)")
print("    - Add more hidden layers or neurons")
print("  → Then run: python 04_improved_model.py")
print("    - Learn about CNNs and achieve >99% accuracy!")
print("=" * 60)

plt.show()
