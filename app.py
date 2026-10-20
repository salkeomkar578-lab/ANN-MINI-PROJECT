"""
app.py — Optional Streamlit Demo for Fashion ANN Four Models Project

Run with:  streamlit run app.py

Requirements (in addition to requirements.txt):
  pip install streamlit Pillow

This demo loads pre-trained models from the models/ directory and allows
the user to select Fashion-MNIST test images for multi-model prediction.
"""
import os
import numpy as np

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed.")
    print("Install it with:  pip install streamlit")
    print("Then run:         streamlit run app.py")
    raise SystemExit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    st.error("TensorFlow is required. Install with: pip install tensorflow")
    st.stop()

# ---- Constants ----
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
MODELS_DIR = 'models'
IMG_PIXELS = 784


@st.cache_resource
def load_models():
    """Load pre-trained MLP and CNN models."""
    models_loaded = {}

    mlp_path = os.path.join(MODELS_DIR, 'mlp_fashion.keras')
    if os.path.exists(mlp_path):
        models_loaded['MLP'] = keras.models.load_model(mlp_path)
    else:
        st.warning(f"MLP model not found at {mlp_path}. Run the notebook first.")

    cnn_path = os.path.join(MODELS_DIR, 'cnn_fashion.keras')
    if os.path.exists(cnn_path):
        models_loaded['CNN'] = keras.models.load_model(cnn_path)
    else:
        st.warning(f"CNN model not found at {cnn_path}. Run the notebook first.")

    return models_loaded


@st.cache_resource
def load_som_weights():
    """Load SOM weights."""
    som_path = os.path.join(MODELS_DIR, 'som_weights.npy')
    if os.path.exists(som_path):
        return np.load(som_path)
    return None


@st.cache_data
def load_test_data():
    """Load Fashion-MNIST test data."""
    (_, _), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return X_test, y_test


def find_bmu(weights, x):
    """Find Best Matching Unit for input vector x."""
    diff = weights - x
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
    return bmu_idx


def main():
    st.set_page_config(
        page_title="Fashion ANN Demo",
        page_icon="👗",
        layout="wide"
    )

    st.title("👗 Fashion Product Recognition — Multi-Model Demo")
    st.markdown(
        "This demo uses pre-trained models from the Fashion ANN mini project. "
        "Select a test image to see predictions from MLP, CNN, and SOM."
    )

    # Load resources
    models = load_models()
    som_weights = load_som_weights()
    X_test, y_test = load_test_data()

    if not models:
        st.error(
            "No models found in models/ directory. "
            "Please run the Jupyter notebook first to train and save models."
        )
        return

    # Sidebar — image selection
    st.sidebar.header("Select Test Image")
    img_idx = st.sidebar.slider("Image index", 0, len(X_test) - 1, 0)
    category_filter = st.sidebar.selectbox(
        "Filter by category", ["All"] + CLASS_NAMES
    )

    if category_filter != "All":
        cat_id = CLASS_NAMES.index(category_filter)
        valid_indices = np.where(y_test == cat_id)[0]
        img_idx = st.sidebar.selectbox(
            "Choose from filtered images",
            valid_indices,
            format_func=lambda x: f"Image {x}"
        )

    # Display selected image
    image = X_test[img_idx]
    true_label = y_test[img_idx]

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        st.subheader("Input Image")
        st.image(image, caption=f"True: {CLASS_NAMES[true_label]}", width=150)

    # MLP prediction
    if 'MLP' in models:
        with col2:
            st.subheader("MLP Prediction")
            x_flat = image.astype(np.float32).reshape(1, IMG_PIXELS) / 255.0
            probs = models['MLP'].predict(x_flat, verbose=0)[0]
            pred = np.argmax(probs)
            confidence = probs[pred]

            if pred == true_label:
                st.success(f"**{CLASS_NAMES[pred]}** ({confidence:.1%})")
            else:
                st.error(f"**{CLASS_NAMES[pred]}** ({confidence:.1%})")

            # Show probability distribution
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['green' if i == pred else 'lightgray' for i in range(10)]
            ax.barh(range(10), probs, color=colors)
            ax.set_yticks(range(10))
            ax.set_yticklabels(CLASS_NAMES, fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)
            plt.close()

    # CNN prediction
    if 'CNN' in models:
        with col3:
            st.subheader("CNN Prediction")
            x_cnn = image.astype(np.float32).reshape(1, 28, 28, 1) / 255.0
            probs = models['CNN'].predict(x_cnn, verbose=0)[0]
            pred = np.argmax(probs)
            confidence = probs[pred]

            if pred == true_label:
                st.success(f"**{CLASS_NAMES[pred]}** ({confidence:.1%})")
            else:
                st.error(f"**{CLASS_NAMES[pred]}** ({confidence:.1%})")

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['green' if i == pred else 'lightgray' for i in range(10)]
            ax.barh(range(10), probs, color=colors)
            ax.set_yticks(range(10))
            ax.set_yticklabels(CLASS_NAMES, fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)
            plt.close()

    # SOM assignment
    if som_weights is not None:
        st.subheader("SOM — Best Matching Unit")
        x_flat = image.astype(np.float32).reshape(IMG_PIXELS) / 255.0
        bmu = find_bmu(som_weights, x_flat)
        proto = som_weights[bmu[0], bmu[1]].reshape(28, 28)

        scol1, scol2 = st.columns(2)
        with scol1:
            st.write(f"BMU Position: row={bmu[0]}, col={bmu[1]}")
            st.image(proto, caption="BMU Prototype", width=150, clamp=True)
        with scol2:
            st.write("This is an **unsupervised cluster assignment**, not a class prediction.")

    st.markdown("---")
    st.caption(
        "Models trained on Fashion-MNIST (28x28 grayscale). "
        "This demo uses the official test set images only."
    )


if __name__ == "__main__":
    main()
