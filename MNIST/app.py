"""
MNIST Digit Recognition Web App

A simple web interface to draw digits and get predictions from a trained neural network.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os


# Define the neural network architecture (must match training)
class SimpleNN(nn.Module):
    """Simple fully-connected neural network for MNIST digit recognition."""

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading on every interaction)."""
    model = SimpleNN()

    # Load the trained weights
    model_path = 'models/simple_nn_trained.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first!")
        return None

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint.get('test_accuracy', 'N/A')


def preprocess_image(image_array):
    """
    Preprocess the drawn image to match MNIST format.

    Args:
        image_array: numpy array from canvas (RGBA format)

    Returns:
        torch.Tensor: Preprocessed image ready for model input
    """
    # Convert RGBA to grayscale
    # Canvas has black background (0,0,0) and white strokes (255,255,255)
    if image_array is None:
        return None

    # Extract RGB channels and convert to grayscale
    # Since we're drawing white on black, we can use any single RGB channel
    # (they're all the same for grayscale)
    rgb_image = image_array[:, :, :3]  # Get RGB, ignore alpha

    # Convert to PIL Image (it will handle RGB to grayscale)
    img = Image.fromarray(rgb_image.astype('uint8')).convert('L')

    # Resize to 28x28 (MNIST size)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype('float32') / 255.0

    # Apply MNIST normalization (same as training)
    mean = 0.1307
    std = 0.3081
    img_normalized = (img_array - mean) / std

    # Convert to PyTorch tensor with shape (1, 1, 28, 28)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

    return img_tensor, img_array


def predict_digit(model, image_tensor):
    """
    Make a prediction on the preprocessed image.

    Args:
        model: Trained neural network
        image_tensor: Preprocessed image tensor

    Returns:
        tuple: (predicted_digit, confidence, all_probabilities)
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_digit = probabilities.argmax().item()
        confidence = probabilities[predicted_digit].item()

    return predicted_digit, confidence, probabilities.numpy()


# ============================================================================
# STREAMLIT APP
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Title and description
st.title("🔢 MNIST Digit Recognition")
st.markdown("""
Draw a digit (0-9) in the canvas below and click **Predict** to see what the neural network thinks it is!

The model is a simple fully-connected neural network trained on the MNIST dataset.
""")

# Load the model
model_info = load_model()
if model_info is None:
    st.stop()

model, test_accuracy = model_info

# Display model info
with st.expander("ℹ️ Model Information"):
    st.write(f"**Architecture:** SimpleNN (784 → 128 → 10)")
    st.write(f"**Test Accuracy:** {test_accuracy:.2f}%" if isinstance(test_accuracy, float) else f"**Test Accuracy:** {test_accuracy}")
    st.write(f"**Parameters:** 101,770")

st.markdown("---")

# Initialize session state for canvas reset
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Create two columns for canvas and results
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Draw Here")

    # Create canvas for drawing (key changes when cleared)
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Black background
        stroke_width=20,
        stroke_color="#FFFFFF",  # White pen
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_button = st.button("🎯 Predict", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.canvas_key += 1
            st.rerun()

with col2:
    st.subheader("🎯 Prediction")

    # Make prediction when button is clicked
    if predict_button and canvas_result.image_data is not None:
        # Check if canvas has any drawing
        if np.sum(canvas_result.image_data[:, :, 3]) == 0:
            st.warning("⚠️ Please draw a digit first!")
        else:
            # Preprocess the image
            processed_data = preprocess_image(canvas_result.image_data)
            if processed_data is not None:
                image_tensor, img_array = processed_data

                # Make prediction
                predicted_digit, confidence, probabilities = predict_digit(model, image_tensor)

                # Display results
                st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                st.markdown(f"**Confidence:** {confidence * 100:.1f}%")

                # Progress bar for confidence
                st.progress(confidence)

                st.markdown("---")

                # Show all probabilities
                st.markdown("**All Predictions:**")
                for digit in range(10):
                    prob = probabilities[digit] * 100
                    is_predicted = digit == predicted_digit

                    # Create a bar for each digit
                    bar_color = "🟢" if is_predicted else "⚪"
                    st.write(f"{bar_color} **{digit}:** {prob:.1f}%")
                    st.progress(float(probabilities[digit]))

                # Show processed image - EXPANDED BY DEFAULT for debugging
                with st.expander("🔍 View Processed Image (28×28)", expanded=True):
                    col_img1, col_img2 = st.columns(2)

                    with col_img1:
                        st.write("**Raw Canvas (RGB):**")
                        # Show the RGB version before resizing
                        raw_rgb = canvas_result.image_data[:, :, :3]
                        st.image(raw_rgb, width=140, caption="Full canvas RGB")

                    with col_img2:
                        st.write("**Model Input (28×28):**")
                        st.image(img_array, width=140, caption="Resized & normalized")

                    st.caption("The drawing is resized to 28×28 pixels and normalized, just like the MNIST training data.")
                    st.caption(f"Min pixel value: {img_array.min():.2f}, Max: {img_array.max():.2f}")
    else:
        st.info("👈 Draw a digit and click **Predict** to see the results!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Built with Streamlit • Powered by PyTorch</p>
    <p>Model trained on the MNIST dataset (60,000 training samples)</p>
</div>
""", unsafe_allow_html=True)
