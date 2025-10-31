# MNIST Neural Network Learning Journey

Welcome to your hands-on journey into Machine Learning, Neural Networks, and Deep Learning! This plan will guide you through understanding and implementing neural networks to recognize handwritten digits from the MNIST dataset.

## Course Reference
This plan is inspired by Michael Nielsen's excellent book: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)

## Progress Tracker
Mark your progress as you complete each phase:

- [ ] Phase 1: Understanding the Basics
- [ ] Phase 2: Your First Neural Network
- [ ] Phase 3: Training & Optimization
- [ ] Phase 4: Improving Performance
- [ ] Phase 5: Going Deeper with CNNs
- [ ] Phase 6: Understanding What You Built

---

## Phase 1: Understanding the Basics

### What You'll Learn
- What is a neural network and why do we need them?
- What is the MNIST dataset?
- How do computers "see" images?
- Basic terminology: neurons, layers, weights, biases

### Concepts (Intuitive Understanding)

**Neural Networks are Function Approximators**
Think of a neural network as a sophisticated pattern matcher. Just like you learned to recognize digits as a child by seeing many examples, a neural network learns patterns from data.

**The MNIST Dataset**
- 70,000 images of handwritten digits (0-9)
- Each image is 28x28 pixels (784 pixels total)
- Each pixel has a value from 0 (black) to 255 (white)
- The network's job: look at 784 numbers and output which digit (0-9) it represents

**How Computers "See" Images**
- Images are just grids of numbers
- For MNIST: 28x28 = 784 numbers representing pixel brightness
- The neural network takes these 784 numbers as input
- It outputs 10 numbers (probabilities for digits 0-9)

### Hands-On Exercise
```bash
python 01_explore_mnist.py
```

**What This Does:**
- Downloads the MNIST dataset
- Shows you sample images
- Displays the shape and structure of the data
- Helps you visualize what you're working with

### Key Questions to Answer
1. What does a handwritten "7" look like as numbers?
2. Why might some digits be harder to classify than others?
3. How would YOU explain to a computer how to recognize a "0" vs an "8"?

**Checkpoint**: You understand that images are just numbers, and you've seen what MNIST data looks like.

---

## Phase 2: Your First Neural Network

### What You'll Learn
- The architecture of a simple neural network
- What are layers, neurons, weights, and biases?
- Activation functions (ReLU, Softmax)
- Forward propagation (making predictions)

### Concepts (Intuitive Understanding)

**Network Architecture**
```
Input Layer (784 neurons)  →  Hidden Layer (128 neurons)  →  Output Layer (10 neurons)
    [pixel values]              [pattern detectors]            [digit probabilities]
```

**What Each Part Does:**

1. **Input Layer (784 neurons)**
   - One neuron per pixel
   - Simply holds the pixel values
   - No computation happens here

2. **Hidden Layer (128 neurons)**
   - Each neuron looks at ALL 784 input pixels
   - Each has its own "weights" - how much attention to pay to each pixel
   - Acts like a pattern detector (e.g., "is there a curve here?", "is there a vertical line?")
   - After weighted sum, applies ReLU activation (keeps positive values, zeros out negative)

3. **Output Layer (10 neurons)**
   - One neuron per digit (0-9)
   - Each looks at all 128 hidden neurons
   - Applies Softmax to convert to probabilities (all add up to 1.0)
   - Highest probability = predicted digit

**Weights and Biases**
- **Weights**: How much each input matters to this neuron (learned during training)
- **Biases**: A baseline adjustment for each neuron (also learned)
- Think of weights as "sensitivity" and biases as "threshold"

**Activation Functions**

**Overview:**
Activation functions introduce non-linearity into neural networks. Without them, multiple layers would collapse into a single linear transformation, unable to learn complex patterns.

**Why Non-linearity Matters:**
```
Without activation: Layer 1 → Layer 2
  y = W1·x + b1
  z = W2·y + b2 = W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)

Result: Just another linear function! Can't learn curves, edges, or complex patterns.
```

---

### Deep Dive: ReLU (Rectified Linear Unit)

**Mathematical Definition:**
```
ReLU(x) = max(0, x)
```

Simply: If input is positive, pass it through. If negative, output zero.

**Examples:**
- ReLU(5) = 5
- ReLU(-3) = 0
- ReLU(0.5) = 0.5
- ReLU(-0.1) = 0

**Visual Representation:**
```
 f(x) |     ╱
      |    ╱
      |   ╱
      |  ╱
      | ╱
──────┼────────── x
      |
     0
```

**Why ReLU is the Default Choice:**

1. **Computationally Efficient**
   - Simple max(0, x) operation
   - Much faster than sigmoid (1/(1+e^-x)) or tanh
   - No expensive exponential calculations

2. **Solves Vanishing Gradient Problem**
   - Older functions (sigmoid, tanh) have very small gradients in deep networks
   - ReLU's gradient is either 0 or 1 - no vanishing for positive values!
   - Enables training of much deeper networks

3. **Sparse Activation**
   - About 50% of neurons output exactly 0 (for negative inputs)
   - Sparsity makes networks more efficient and interpretable
   - Helps prevent overfitting

4. **Biological Plausibility**
   - Similar to real neurons: either fire or don't fire
   - More realistic than smooth sigmoid curves

**Potential Drawbacks:**

1. **"Dying ReLU" Problem**
   - If a neuron always outputs negative, it always returns 0
   - Gradient is 0, so weights never update (neuron "dies")
   - Can happen with high learning rates or poor initialization
   - **Solutions**: Leaky ReLU variants, careful learning rate, proper initialization

2. **Not Zero-Centered**
   - Outputs are always ≥ 0
   - Can make optimization slightly less efficient
   - Usually not a major issue in practice

**Comparison with Other Activation Functions:**

| Function | Formula | Range | Gradient | Common Use |
|----------|---------|-------|----------|------------|
| **ReLU** | max(0,x) | [0, ∞) | 0 or 1 | Hidden layers (default choice) |
| Sigmoid | 1/(1+e^-x) | (0, 1) | Very small | Output (binary classification) |
| Tanh | (e^x-e^-x)/(e^x+e^-x) | (-1, 1) | Larger than sigmoid | Hidden (less common now) |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | 0.01 or 1 | Fixes dying ReLU |
| **Softmax** | e^xi/Σe^xj | (0,1), sum=1 | Various | Output (multi-class) |

**How ReLU Works in Your Hidden Layer:**

For each of the 128 hidden neurons:
1. Calculate weighted sum: `z = (w1·pixel1 + w2·pixel2 + ... + w784·pixel784) + bias`
2. Apply ReLU: `activation = max(0, z)`
3. If z > 0: neuron "fires" and passes signal forward (pattern detected!)
4. If z ≤ 0: neuron stays silent (pattern not detected)

**Key Insight:** Each neuron acts like a pattern detector. ReLU lets it "speak up" when it finds its pattern (positive output) or stay quiet when it doesn't (zero output). About half your neurons will be "off" at any time, creating an efficient, sparse representation.

**Experiment Idea:** Try removing ReLU (just use `activation = z` without max) in your network. You'll find it can't learn to distinguish digits no matter how many layers you add - proving that non-linearity is essential!

---

### Deep Dive: Softmax Activation (Output Layer)

**Mathematical Definition:**

For a vector of raw scores (logits) [z₁, z₂, ..., zₙ], Softmax converts them to probabilities:

```
Softmax(zᵢ) = e^zᵢ / (e^z₁ + e^z₂ + ... + e^zₙ)
```

In other words: Take the exponential of each score, then divide by the sum of all exponentials.

**Step-by-Step Example (MNIST Output Layer):**

Imagine your network outputs these raw scores for an image:

```
Raw scores (logits):
  Digit 0: 0.5
  Digit 1: 2.0
  Digit 2: 0.3
  Digit 3: 0.1
  Digit 4: 0.2
  Digit 5: 0.4
  Digit 6: 0.2
  Digit 7: 3.5  ← highest score
  Digit 8: 0.6
  Digit 9: 0.3
```

**Step 1: Exponentiate each score**
```
e^0.5 = 1.65,  e^2.0 = 7.39,  e^0.3 = 1.35, e^0.1 = 1.11,  e^0.2 = 1.22
e^0.4 = 1.49,  e^0.2 = 1.22,  e^3.5 = 33.12, e^0.6 = 1.82, e^0.3 = 1.35
```

**Step 2: Sum all exponentials**
```
Sum = 1.65 + 7.39 + 1.35 + 1.11 + 1.22 + 1.49 + 1.22 + 33.12 + 1.82 + 1.35 = 51.72
```

**Step 3: Divide each by the sum**
```
Probabilities after Softmax:
  Digit 0: 1.65/51.72 = 0.032 (3.2%)
  Digit 1: 7.39/51.72 = 0.143 (14.3%)
  Digit 2: 1.35/51.72 = 0.026 (2.6%)
  Digit 3: 1.11/51.72 = 0.021 (2.1%)
  Digit 4: 1.22/51.72 = 0.024 (2.4%)
  Digit 5: 1.49/51.72 = 0.029 (2.9%)
  Digit 6: 1.22/51.72 = 0.024 (2.4%)
  Digit 7: 33.12/51.72 = 0.640 (64.0%) ← Network predicts "7"!
  Digit 8: 1.82/51.72 = 0.035 (3.5%)
  Digit 9: 1.35/51.72 = 0.026 (2.6%)

Total = 100% ✓
```

**Visual Transformation:**

```
Raw Scores          Softmax         Probabilities
(logits)            ────→           (0 to 1, sum=1)

     3.5 │█████                      0.64 │████████████████
         │                                │
     2.0 │███                        0.14 │███
         │                                │
     1.0 │█                          0.03 │█
         │                                │
     0.0 ├──────                     0.00 ├──────
         0 1 2 3 4 5 6 7 8 9              0 1 2 3 4 5 6 7 8 9
```

**Why Softmax for Multi-Class Classification?**

1. **Produces Valid Probabilities**
   - All outputs are between 0 and 1
   - All outputs sum to exactly 1.0
   - You can interpret them as "confidence" for each class

2. **Amplifies Differences**
   - Exponential function amplifies gaps between scores
   - High scores become much higher probabilities
   - Low scores get suppressed to near-zero
   - Makes predictions more confident and decisive

3. **Differentiable**
   - Smooth gradient for backpropagation
   - Network can learn to adjust scores during training

4. **Theoretically Grounded**
   - Softmax is the multi-class generalization of sigmoid
   - Arises naturally from maximum entropy principles
   - Matches multinomial logistic regression

**Key Properties:**

| Property | Description | Example |
|----------|-------------|---------|
| **Range** | Each output in (0, 1) | Never exactly 0 or 1 |
| **Sum** | All outputs sum to 1.0 | Can interpret as probabilities |
| **Monotonic** | Higher input → higher output | Preserves ordering |
| **Amplification** | Exaggerates differences | Score of 3 vs 2 → 10x probability difference |

**Softmax vs Other Output Activations:**

| Task | Output Activation | Output Shape | Loss Function |
|------|------------------|--------------|---------------|
| **Multi-class (MNIST)** | **Softmax** | [p₀, p₁, ..., p₉], sum=1 | Cross-Entropy |
| Binary classification | Sigmoid | Single value in (0, 1) | Binary Cross-Entropy |
| Regression | None (linear) | Raw value in (-∞, ∞) | MSE |
| Multi-label | Sigmoid (per class) | Independent probabilities | Binary Cross-Entropy |

**Temperature in Softmax (Advanced):**

Softmax can be adjusted with a "temperature" parameter T:

```
Softmax_T(zᵢ) = e^(zᵢ/T) / Σ e^(zⱼ/T)
```

- **T = 1**: Standard Softmax (what you're using)
- **T → 0**: Becomes "hard max" (one-hot, winner takes all)
- **T → ∞**: Becomes uniform distribution (all equal)
- **T < 1**: Sharper, more confident predictions
- **T > 1**: Softer, less confident predictions

This is useful for model distillation and calibration (advanced topics).

**How Softmax Works in Your Network:**

In your MNIST network's output layer:

1. **Input**: 128 activations from hidden layer
2. **Weighted sum**: Each of 10 output neurons computes its score
   - `score_digit_0 = w₁·h₁ + w₂·h₂ + ... + w₁₂₈·h₁₂₈ + bias₀`
   - `score_digit_1 = ...`
   - ... (repeat for all 10 digits)
3. **Softmax**: Convert 10 raw scores to 10 probabilities
4. **Prediction**: `argmax(probabilities)` = digit with highest probability
5. **Training**: Compare probabilities to true label using Cross-Entropy Loss

**Key Insight:** Softmax turns your network's raw opinions (logits) into a well-calibrated probability distribution. When it outputs [0.02, 0.14, 0.03, 0.02, 0.02, 0.03, 0.02, 0.64, 0.04, 0.03], it's saying: "I'm 64% confident this is a 7, 14% confident it's a 1, and much less confident about other digits."

**Common Misconception:** Softmax doesn't just normalize scores by dividing by their sum. The exponential is crucial - it amplifies differences, making the network more decisive. Compare:

```
Scores: [2.0, 3.5, 1.0]

Simple normalization: [2.0/6.5, 3.5/6.5, 1.0/6.5] = [0.31, 0.54, 0.15]
Softmax:              [7.39/45.9, 33.1/45.9, 2.72/45.9] = [0.16, 0.72, 0.06]
                       ↑ More decisive! Highest class gets emphasized
```

**Why Not Just Use argmax()?**

You might wonder: "Why not just pick the highest score?"

During training, we need:
- **Gradients**: argmax has no gradient (it's discrete), Softmax is smooth
- **Loss signal**: Cross-Entropy Loss needs probabilities to measure "how wrong"
- **Confidence**: Probabilities tell us how certain the network is

### Hands-On Exercise
```bash
python 02_simple_nn.py
```

**What This Does:**
- Defines a simple neural network architecture
- Shows the network structure and parameter count
- Makes a prediction on a random digit (before training - will be random!)
- Helps you understand the network shape

### Key Questions to Answer
1. How many parameters (weights + biases) does your network have?
2. Why are the initial predictions random?
3. What would happen if we removed the activation functions?

**Checkpoint**: You can create a neural network and make predictions (even if they're wrong initially).

---

## Phase 3: Training & Optimization

### What You'll Learn
- Loss functions (how to measure error)
- Gradient descent (how to improve)
- Backpropagation (the intuition, not the math)
- Training, validation, and test sets
- Epochs, batches, learning rate

### Concepts (Intuitive Understanding)

**The Training Process**
Think of training like learning to throw darts:
1. Throw a dart (make a prediction)
2. See how far off you were (calculate loss)
3. Adjust your aim (update weights)
4. Repeat until you consistently hit the target

**Loss Function (Cross-Entropy)**
- Measures how wrong your predictions are
- Low loss = good predictions, high loss = bad predictions
- The network's goal: minimize this loss

**Gradient Descent**
Imagine you're blindfolded on a hill and want to reach the valley:
- Feel which direction is downhill (calculate gradients)
- Take a step in that direction (update weights)
- Learning rate = size of each step
  - Too large: you might overshoot the valley
  - Too small: takes forever to get down

**Backpropagation (The Intuition)**
- After making a prediction, we know the error
- Backpropagation figures out: "which weights caused this error?"
- Works backward through the network
- Adjusts weights proportional to their contribution to the error
- You don't need to code this - PyTorch does it automatically!

**Data Splits**
- **Training Set (80%)**: Used to train the network
- **Validation Set (10%)**: Used to check progress and tune hyperparameters
- **Test Set (10%)**: Final evaluation - only use once at the end!

**Key Hyperparameters**
- **Epochs**: How many times to go through the entire training set
- **Batch Size**: How many examples to process before updating weights
  - Smaller batches: noisier but more frequent updates
  - Larger batches: smoother but fewer updates
- **Learning Rate**: How big each weight update should be

### Hands-On Exercise
```bash
python 03_train_model.py
```

**What This Does:**
- Trains your neural network on MNIST
- Shows loss decreasing over time
- Displays accuracy on validation set
- Saves the trained model
- Shows training progress with real-time metrics

### What to Watch For
1. **Loss should decrease**: If it doesn't, something's wrong
2. **Accuracy should increase**: Aiming for ~97-98% on validation
3. **Training vs Validation**: If training accuracy >> validation accuracy, you're overfitting

### Key Questions to Answer
1. How does the loss change as training progresses?
2. What accuracy can you achieve after 10 epochs?
3. What happens if you change the learning rate (try 0.1, 0.01, 0.001)?

**Checkpoint**: You've trained a neural network and can see it learning!

---

## Phase 4: Improving Performance

### What You'll Learn
- Techniques to improve accuracy
- Regularization (preventing overfitting)
- Hyperparameter tuning
- Common problems and solutions

### Concepts (Intuitive Understanding)

**Overfitting vs Underfitting**
- **Overfitting**: Network memorizes training data but doesn't generalize
  - Like memorizing answers instead of understanding concepts
  - High training accuracy, low validation accuracy
- **Underfitting**: Network hasn't learned enough
  - Like giving up too early while studying
  - Low accuracy on both training and validation

**Regularization Techniques**

1. **Dropout**: Randomly "turn off" some neurons during training
   - Prevents over-reliance on specific neurons
   - Forces network to learn robust features
   - Like practicing with different teammates

2. **Batch Normalization**: Normalizes inputs to each layer
   - Makes training more stable and faster
   - Reduces sensitivity to initialization

3. **Early Stopping**: Stop training when validation loss stops improving
   - Prevents overfitting
   - Automatic way to choose number of epochs

**Hyperparameter Tuning**
Things to experiment with:
- Number of hidden layers (try 1, 2, 3)
- Number of neurons per layer (try 64, 128, 256, 512)
- Learning rate (try 0.1, 0.01, 0.001, 0.0001)
- Batch size (try 32, 64, 128, 256)
- Dropout rate (try 0.2, 0.3, 0.5)

**Optimization Algorithms**
- **SGD**: Basic gradient descent
- **Adam**: Adaptive learning rate (usually better for beginners)
  - Adjusts learning rate automatically for each parameter
  - More forgiving to hyperparameter choices

### Hands-On Exercise
Modify `03_train_model.py` to experiment with:
1. Adding more hidden layers
2. Changing layer sizes
3. Adding dropout
4. Trying different optimizers (SGD vs Adam)
5. Adjusting learning rate

### Key Questions to Answer
1. What combination of hyperparameters gives you the best accuracy?
2. Can you get above 98% accuracy with a simple fully-connected network?
3. How does adding dropout affect training vs validation accuracy?

**Checkpoint**: You understand how to tune a network and can achieve ~98% accuracy.

---

## Phase 5: Going Deeper with CNNs

### What You'll Learn
- Why Convolutional Neural Networks (CNNs) work better for images
- Convolutions, pooling, and feature maps
- Building a CNN in PyTorch
- Achieving state-of-the-art results (>99% accuracy)

### Concepts (Intuitive Understanding)

**Why CNNs for Images?**

**Problem with Fully Connected Networks:**
- Treats pixels independently
- Doesn't understand spatial relationships
- "A curve at the top-left" is different from "a curve at the bottom-right"
- Too many parameters (784 × 128 = 100,352 weights just for first layer!)

**How CNNs Solve This:**

1. **Convolution Layers**: Use small filters (e.g., 3×3) that scan across the image
   - Same filter weights are reused across the entire image
   - Learns to detect features like edges, curves, corners
   - Fewer parameters, more efficient
   - Preserves spatial relationships

2. **Pooling Layers**: Downsample the image
   - Reduces size (e.g., 28×28 → 14×14)
   - Makes network more robust to small shifts
   - "Max pooling" keeps the strongest signal in each region

3. **Multiple Feature Maps**: Learn multiple filters
   - Different filters detect different features
   - Early layers: simple features (edges, corners)
   - Deeper layers: complex features (digit parts, shapes)

**CNN Architecture for MNIST**
```
Input (28×28)
    ↓
Conv Layer (32 filters, 3×3) + ReLU
    ↓
Max Pooling (2×2)  [now 14×14]
    ↓
Conv Layer (64 filters, 3×3) + ReLU
    ↓
Max Pooling (2×2)  [now 7×7]
    ↓
Flatten  [7×7×64 = 3,136 values]
    ↓
Fully Connected (128) + ReLU + Dropout
    ↓
Output (10) + Softmax
```

**Why This Works Better:**
- Fewer parameters than fully connected (more efficient)
- Understands spatial structure (edges, shapes)
- Translation invariant (recognizes "7" anywhere in the image)
- Hierarchical feature learning (simple → complex)

### Hands-On Exercise
```bash
python 04_improved_model.py
```

**What This Does:**
- Implements a CNN architecture
- Trains on MNIST
- Compares performance with simple network
- Visualizes what filters learn (feature maps)
- Should achieve >99% accuracy!

### Key Questions to Answer
1. How does CNN accuracy compare to your simple network?
2. How many parameters does the CNN have vs the fully connected network?
3. What features do the first convolutional layer filters detect?

**Checkpoint**: You understand why CNNs work better for images and can build one.

---

## Phase 6: Understanding What You Built

### What You'll Learn
- Visualizing what the network learned
- Understanding mistakes and failure modes
- Feature visualization and interpretation
- Common pitfalls in ML
- Next steps in your ML journey

### Visualization Exercises

**1. Visualize Predictions**
- Look at correct predictions: What patterns does the network see?
- Look at wrong predictions: Why did it fail?
- Common confusions: 4/9, 3/5, 7/1

**2. Visualize Learned Features**
- First conv layer: Typically learns edge detectors
- Second conv layer: Learns more complex patterns
- What features matter most for each digit?

**3. Activation Maps**
- Which parts of the image activate each neuron?
- What is the network "paying attention to"?

**4. Decision Boundaries**
- What makes the network confident vs uncertain?
- Can you find adversarial examples (images that fool the network)?

### Common Pitfalls and Lessons

**1. Data Leakage**
- Never train on test data!
- Use validation set for hyperparameter tuning
- Test set only for final evaluation

**2. Overfitting**
- High training accuracy but low validation accuracy
- Solutions: dropout, more data, simpler model, early stopping

**3. Vanishing/Exploding Gradients**
- Very deep networks can be hard to train
- Solutions: ReLU activations, batch normalization, skip connections

**4. Poor Initialization**
- Bad weight initialization can prevent learning
- PyTorch handles this automatically for standard layers

**5. Wrong Learning Rate**
- Too high: Loss explodes or oscillates
- Too low: Slow learning or getting stuck
- Solution: Start with 0.001 (for Adam) and adjust

### Reflection Questions

**Understanding:**
1. Can you explain how a neural network makes a prediction to a friend?
2. Why does backpropagation work?
3. Why are CNNs better than fully connected networks for images?

**Practical:**
4. What accuracy did you achieve with your best model?
5. Which hyperparameters had the biggest impact?
6. What digits does your network confuse most often?

**Deeper Thinking:**
7. What real-world problems could you solve with similar techniques?
8. What are the limitations of what you built?
9. How would you handle colored images? Different sizes?

---

## Next Steps in Your ML Journey

### Immediate Next Steps
1. **Experiment More with MNIST**
   - Try different architectures
   - Implement data augmentation (rotate, shift images)
   - Try ensemble methods (combine multiple models)

2. **Try Other Datasets**
   - Fashion-MNIST (clothing items instead of digits)
   - CIFAR-10 (color images, 10 classes)
   - Your own image dataset

3. **Learn More Techniques**
   - Transfer learning
   - Data augmentation
   - Learning rate scheduling
   - Different optimizers

### Learning Resources

**Books:**
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen (free online)
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Deep Learning with Python" by François Chollet

**Online Courses:**
- Fast.ai - Practical Deep Learning for Coders
- Andrew Ng's Machine Learning Course (Coursera)
- PyTorch Tutorials (official documentation)

**Practice:**
- Kaggle competitions (start with tutorials)
- Papers with Code (implement research papers)
- Build projects with real-world data

### Advanced Topics to Explore
- Recurrent Neural Networks (RNNs) for sequences
- Transformers for NLP
- Generative Adversarial Networks (GANs)
- Reinforcement Learning
- Computer Vision applications
- Natural Language Processing

---

## Project Structure

```
MNIST/
├── learning_plan.md           # This file
├── requirements.txt           # Python dependencies
├── 01_explore_mnist.py        # Phase 1: Explore the dataset
├── 02_simple_nn.py            # Phase 2: Build your first network
├── 03_train_model.py          # Phase 3: Train and evaluate
├── 04_improved_model.py       # Phase 5: CNN implementation
├── models/                    # Saved model checkpoints
└── visualizations/            # Generated plots and visualizations
```

---

## Tips for Success

1. **Go at your own pace** - Don't rush. Understanding is more important than speed.

2. **Experiment actively** - Don't just run the code. Change things and see what happens!

3. **Debug with prints** - Add print statements to see shapes, values, and intermediate results.

4. **Visualize everything** - Plot losses, accuracies, predictions, and learned features.

5. **Ask "why?"** - Don't just accept that something works. Understand why.

6. **Make mistakes** - You'll learn more from debugging than from code that works first try.

7. **Track your experiments** - Write down what you try and what results you get.

8. **Take breaks** - Let concepts sink in. Come back with fresh eyes.

---

## Notes Section

Use this space to track your progress, insights, and questions:

### My Progress Notes

**Date: ______**
- Completed phases:
- Current accuracy:
- Best hyperparameters found:
- Interesting observations:
- Questions to explore:

---

## Troubleshooting

### Installation Issues
If you have trouble installing PyTorch, visit: https://pytorch.org/get-started/locally/
Select your OS and preferences to get the correct installation command.

### Code Issues
- Check tensor shapes with `.shape`
- Verify data is on the correct device (CPU vs GPU)
- Make sure your data is normalized (0-1 range for MNIST)
- Check that you're in train vs eval mode

### Performance Issues
- If accuracy isn't improving: check learning rate, verify data preprocessing
- If loss is NaN: learning rate too high or numerical instability
- If training is slow: consider using GPU, reduce batch size, or simplify model

---

Happy Learning! Remember: Every expert was once a beginner. Take it step by step, and you'll be amazed at what you can build.
