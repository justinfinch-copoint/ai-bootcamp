# Quick Reference Guide - Neural Networks Concepts

A handy reference for key concepts you'll encounter in this project.

## Core Concepts

### Neural Network Architecture

| Component | Purpose | Example |
|-----------|---------|---------|
| **Input Layer** | Receives raw data | 784 neurons (28×28 pixels) |
| **Hidden Layer** | Learns patterns/features | 128 neurons |
| **Output Layer** | Makes predictions | 10 neurons (digits 0-9) |
| **Weights** | Connection strengths (learned) | ~100,000 parameters |
| **Biases** | Neuron thresholds (learned) | One per neuron |

### Activation Functions

| Function | Formula | Purpose | When to Use |
|----------|---------|---------|-------------|
| **ReLU** | `max(0, x)` | Introduces non-linearity | Hidden layers (most common) |
| **Softmax** | `e^x / Σe^x` | Converts to probabilities | Output layer (classification) |
| **Sigmoid** | `1 / (1 + e^-x)` | Maps to 0-1 range | Binary classification |

### Training Components

| Component | Purpose | Typical Value |
|-----------|---------|---------------|
| **Loss Function** | Measures prediction error | CrossEntropyLoss |
| **Optimizer** | Updates weights | Adam (lr=0.001) |
| **Learning Rate** | Step size for updates | 0.001 - 0.01 |
| **Batch Size** | Samples per update | 32 - 256 |
| **Epochs** | Full passes through data | 10 - 50 |

### CNN-Specific Concepts

| Component | Purpose | Example |
|-----------|---------|---------|
| **Convolution** | Detects local patterns | 3×3 filter |
| **Pooling** | Downsamples feature maps | 2×2 max pooling |
| **Feature Maps** | Detected patterns | 32 or 64 channels |
| **Padding** | Preserves size | padding=1 |
| **Stride** | Step size for filter | stride=2 |

## Common Hyperparameters

### For Simple Neural Networks

```python
# Architecture
input_size = 784        # 28×28 pixels
hidden_size = 128       # Try: 64, 128, 256, 512
output_size = 10        # 10 digit classes

# Training
learning_rate = 0.001   # Try: 0.1, 0.01, 0.001, 0.0001
batch_size = 64         # Try: 32, 64, 128, 256
epochs = 10             # Try: 5, 10, 20, 50
```

### For CNNs

```python
# Architecture
conv1_channels = 32     # Try: 16, 32, 64
conv2_channels = 64     # Try: 32, 64, 128
kernel_size = 3         # Try: 3, 5, 7
dropout_rate = 0.5      # Try: 0.2, 0.3, 0.5

# Training
learning_rate = 0.001
batch_size = 64
epochs = 12
```

## Performance Metrics

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Accuracy** | `correct / total × 100` | % of correct predictions |
| **Loss** | `criterion(pred, true)` | How wrong predictions are (lower is better) |
| **Precision** | `TP / (TP + FP)` | Of predicted positives, % correct |
| **Recall** | `TP / (TP + FN)` | Of actual positives, % found |

## Common Issues & Solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Overfitting** | Train acc >> Val acc | Add dropout, more data, less complexity |
| **Underfitting** | Low accuracy on both | More complex model, train longer |
| **Exploding Gradients** | Loss becomes NaN | Lower learning rate, gradient clipping |
| **Vanishing Gradients** | No learning progress | Use ReLU, batch normalization |
| **Slow Training** | Takes forever | Increase batch size, use GPU, simpler model |

## Data Splits

| Split | Size | Purpose | When to Use |
|-------|------|---------|-------------|
| **Training** | 60-80% | Learn patterns | Every epoch |
| **Validation** | 10-20% | Tune hyperparameters | Every epoch (for monitoring) |
| **Test** | 10-20% | Final evaluation | Once at the very end |

**Golden Rule**: Never train on test data!

## PyTorch Quick Reference

### Creating Layers

```python
# Fully connected (dense)
fc = nn.Linear(input_size, output_size)

# Convolutional
conv = nn.Conv2d(in_channels, out_channels, kernel_size)

# Pooling
pool = nn.MaxPool2d(kernel_size, stride)

# Dropout
dropout = nn.Dropout(p=0.5)
```

### Training Loop Pattern

```python
model.train()  # Set to training mode
for images, labels in train_loader:
    optimizer.zero_grad()      # Clear gradients
    outputs = model(images)    # Forward pass
    loss = criterion(outputs, labels)  # Calculate loss
    loss.backward()            # Backward pass (backprop)
    optimizer.step()           # Update weights
```

### Evaluation Pattern

```python
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Don't compute gradients
    for images, labels in test_loader:
        outputs = model(images)
        # Calculate accuracy, loss, etc.
```

## Expected Results

### Phase 3: Simple Neural Network
- **Architecture**: 784 → 128 → 10
- **Parameters**: ~101,000
- **Expected Accuracy**: 97-98%
- **Training Time**: ~1-2 minutes (CPU)

### Phase 5: CNN
- **Architecture**: Conv → Pool → Conv → Pool → FC
- **Parameters**: ~110,000
- **Expected Accuracy**: >99%
- **Training Time**: ~3-5 minutes (CPU)

## Debugging Checklist

When things don't work:

- [ ] Check data shapes (use `.shape`)
- [ ] Verify data is normalized
- [ ] Ensure model is in correct mode (`.train()` or `.eval()`)
- [ ] Check learning rate (not too high/low)
- [ ] Verify loss is decreasing
- [ ] Check for NaN in loss
- [ ] Ensure data and model on same device (CPU/GPU)
- [ ] Look at sample predictions
- [ ] Visualize training curves

## Experimentation Ideas

### Easy Experiments
1. Change number of hidden neurons (64, 256, 512)
2. Try different learning rates (0.01, 0.001, 0.0001)
3. Adjust batch size (32, 128, 256)
4. Train for more/fewer epochs

### Medium Experiments
5. Add another hidden layer
6. Try different optimizers (SGD, Adam, RMSprop)
7. Add dropout with different rates
8. Try different activation functions

### Advanced Experiments
9. Implement learning rate scheduling
10. Add batch normalization
11. Try different CNN architectures
12. Implement data augmentation
13. Create an ensemble of models

## Mathematical Notation (Optional)

If you see these in resources:

| Symbol | Meaning |
|--------|---------|
| **x** | Input vector/matrix |
| **y** | True label/output |
| **ŷ** | Predicted output |
| **W** | Weight matrix |
| **b** | Bias vector |
| **σ** | Activation function |
| **L** | Loss function |
| **η** (eta) | Learning rate |
| **∇** (nabla) | Gradient |

## Useful Commands

```bash
# Install requirements
pip install -r requirements.txt

# Run scripts in order
python 01_explore_mnist.py
python 02_simple_nn.py
python 03_train_model.py
python 04_improved_model.py

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check if GPU available
python -c "import torch; print(torch.cuda.is_available())"
```

## Further Reading

- **Book**: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- **PyTorch Docs**: https://pytorch.org/docs/
- **Tutorials**: https://pytorch.org/tutorials/
- **CS231n**: http://cs231n.stanford.edu/ (Stanford's CNN course)

---

**Pro Tip**: Keep this file open while working through the project for quick reference!
