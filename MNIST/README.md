# MNIST Neural Network Learning Project

Welcome to your hands-on journey into Machine Learning and Deep Learning! This project will teach you how neural networks work by implementing digit recognition from the MNIST dataset.

## Quick Start

### 1. Install Dependencies

```bash
cd MNIST
pip install -r requirements.txt
```

### 2. Follow the Learning Path

Run the scripts in order:

```bash
# Phase 1: Explore the data
python 01_explore_mnist.py

# Phase 2: Build your first network
python 02_simple_nn.py

# Phase 3: Train the network (the magic happens here!)
python 03_train_model.py

# Phase 5: Advanced CNN model (>99% accuracy)
python 04_improved_model.py
```

### 3. Test Your Trained Model (Interactive Web App)

After completing Phase 3 (training), you can test your model interactively:

```bash
streamlit run app.py
```

This launches a web application where you can:
- Draw digits (0-9) on a canvas with your mouse
- Get real-time predictions from your trained model
- See confidence scores and probabilities for all digits
- View how your drawing is preprocessed to 28×28 pixels

The app will automatically load your trained model from `models/simple_nn_trained.pth`.

### 4. Read the Learning Plan

For detailed explanations of concepts, theory, and next steps, read:
- **[learning_plan.md](learning_plan.md)** - Your comprehensive guide

## Project Structure

```
MNIST/
├── README.md                  # This file - quick start guide
├── learning_plan.md           # Detailed learning plan with concepts
├── requirements.txt           # Python dependencies
├── 01_explore_mnist.py        # Phase 1: Explore dataset
├── 02_simple_nn.py            # Phase 2: First neural network
├── 03_train_model.py          # Phase 3: Training
├── 04_improved_model.py       # Phase 5: CNN implementation
├── app.py                     # Interactive web app for testing models
├── models/                    # Saved trained models
├── visualizations/            # Generated plots and images
└── data/                      # MNIST dataset (auto-downloaded)
```

## Expected Results

- **Simple Neural Network** (Phase 3): ~97-98% accuracy
- **Convolutional Neural Network** (Phase 5): >99% accuracy

## What You'll Learn

1. **Neural Network Fundamentals**
   - Layers, neurons, weights, biases
   - Activation functions (ReLU, Softmax)
   - Forward propagation

2. **Training Process**
   - Loss functions (Cross-Entropy)
   - Backpropagation (intuitive understanding)
   - Gradient descent and optimization
   - Train/validation/test splits

3. **Advanced Techniques**
   - Convolutional Neural Networks (CNNs)
   - Data augmentation
   - Dropout regularization
   - Learning rate scheduling

4. **Practical Skills**
   - PyTorch framework
   - Data loading and preprocessing
   - Model training and evaluation
   - Visualization and interpretation
   - Deploying models to interactive web applications

## Tips for Success

- **Go in order** - Each phase builds on the previous one
- **Read the code comments** - They explain what's happening
- **Experiment!** - Change parameters and see what happens
- **Visualize** - Check the generated images in `visualizations/`
- **Use the learning plan** - Refer to `learning_plan.md` for detailed explanations

## Troubleshooting

### Installation Issues

If PyTorch installation fails, visit https://pytorch.org/get-started/locally/ and select your platform to get the correct command.

### Common Issues

- **CUDA/GPU errors**: The code works on CPU, GPU is optional
- **Memory errors**: Reduce batch_size in the training scripts
- **Import errors**: Make sure you installed all requirements

## Resources

- [Neural Networks and Deep Learning Book](http://neuralnetworksanddeeplearning.com/chap1.html) - Excellent free resource
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch docs
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/) - Original dataset page

## Next Steps After Completing This Project

1. Try other datasets (Fashion-MNIST, CIFAR-10)
2. Experiment with different architectures
3. Learn about modern architectures (ResNet, Vision Transformers)
4. Explore other domains (NLP, Reinforcement Learning)

---

**Happy Learning!** Remember: every expert was once a beginner. Take your time, experiment, and enjoy the journey of understanding how neural networks learn!
