# AI-Based Fashion Product Recognition and Pattern Discovery Using Four ANN Models

**GitHub Repository**: [https://github.com/salkeomkar578-lab/ANN-MINI-PROJECT](https://github.com/salkeomkar578-lab/ANN-MINI-PROJECT)

## Project Overview

This mini project demonstrates the breadth of **Artificial Neural Network (ANN)** paradigms
by training and evaluating **four distinct models** on the **Fashion-MNIST** dataset:

| # | Model | Paradigm | Objective |
|---|-------|----------|-----------|
| 1 | **Multilayer Perceptron (MLP)** | Supervised | Classify 28×28 grayscale fashion images into 10 known categories |
| 2 | **Convolutional Neural Network (CNN)** | Supervised | Classify the same images using spatial/convolutional feature extraction |
| 3 | **Self-Organizing Map (SOM)** | Unsupervised | Discover patterns and organize images on a 2-D topological map |
| 4 | **Adaptive Resonance Theory 1 (ART1)** | Unsupervised | Learn binary-pattern categories in a stability–plasticity framework |

The project is designed for a **B.Tech AI & Data Science** mini-project submission,
laboratory demonstration, presentation, and viva preparation.

---

## Dataset

**Fashion-MNIST** — 70 000 grayscale images (28×28 px) of fashion products in 10 categories.

- **Source:** <https://github.com/zalandoresearch/fashion-mnist>
- **Paper:** Xiao, Rasul & Vollgraf (2017) — <https://arxiv.org/abs/1708.07747>
- **Loaded via:** `tf.keras.datasets.fashion_mnist.load_data()`

Categories: T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot

---

## Required Software

| Package | Purpose |
|---------|---------|
| Python 3.9 – 3.14 | Runtime |
| TensorFlow ≥ 2.12 | MLP and CNN models |
| NumPy ≥ 1.23 | Numerical computation, SOM & ART1 |
| Matplotlib ≥ 3.6 | Visualization |
| scikit-learn ≥ 1.2 | Evaluation metrics |
| pandas ≥ 1.5 | Tabular display |
| seaborn ≥ 0.12 | Enhanced plots |

---

## Installation

```bash
# 1. (Recommended) Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Google Colab) — dependencies are pre-installed; just upload
#    Fashion_ANN_Four_Models_Project.ipynb and run all cells.
```

---

## How to Run the Notebook

```bash
jupyter notebook Fashion_ANN_Four_Models_Project.ipynb
```

Then select **Kernel → Restart & Run All**.

> **Tip:** On first run the notebook downloads ~30 MB of Fashion-MNIST data.
> Subsequent runs use the cached copy.

---

## Project Structure

```
ANN/
├── Fashion_ANN_Four_Models_Project.ipynb   # Main notebook (complete project, fully executed)
├── PRESENTATION_SCRIPT.md                  # Comprehensive 4-member presentation script & viva guide
├── dataset_loader.py                       # Native Zalando Research gzip dataset reader
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── app.py                                  # Interactive Streamlit multi-model demo
├── models/                                 # Saved pre-trained model weights (MLP, CNN, SOM, ART1)
│   ├── mlp_fashion.keras
│   ├── cnn_fashion.keras
│   ├── som_weights.npy
│   └── art1_weights.npz
└── results/                                # 23 generated plots, curves, and confusion matrices
    ├── mlp_learning_curves.png
    ├── cnn_learning_curves.png
    ├── cnn_feature_maps.png
    ├── som_grid.png
    ├── confusion_matrix_mlp.png
    ├── confusion_matrix_cnn.png
    └── ...
```

---

## The Four Models — At a Glance

### 1. Multilayer Perceptron (MLP)
A classic feed-forward network with fully connected layers. Each 28×28 image is
flattened into a 784-element vector. The network learns to map pixels → class
probabilities through backpropagation.

### 2. Convolutional Neural Network (CNN)
Uses convolutional filters that slide over the image to detect spatial features
(edges, textures, shapes). Preserves the 2-D structure of the image, which
typically improves accuracy over a plain MLP.

### 3. Self-Organizing Map (SOM)
An unsupervised network that projects high-dimensional data onto a 2-D grid.
Nearby neurons learn similar patterns, producing a *topological map* of the
dataset — useful for visualization and cluster exploration.

### 4. Adaptive Resonance Theory 1 (ART1)
A classical unsupervised network for binary patterns. ART1 dynamically creates
categories and uses a *vigilance parameter* to control specificity. It balances
learning new information (plasticity) with preserving old knowledge (stability).

---

## Expected Outputs

After running the full notebook you will obtain:

- **MLP & CNN:** Test accuracy, precision/recall/F1 per class, confusion matrices,
  learning curves, and sample predictions with confidence scores.
- **SOM:** Prototype-image grid, neuron-hit map, class-distribution overlay,
  quantization error, ARI, and NMI.
- **ART1:** Category count vs. vigilance, cluster-size histograms, prototype
  visualizations, ARI, and NMI on held-out data.
- **Comparison tables** contrasting supervised and unsupervised results.

---

## Known Limitations

- Fashion-MNIST images are low resolution (28×28, grayscale). Results do not
  transfer directly to real-world retail photographs.
- SOM training on the full 60 000-image set can be slow on CPU; a configurable
  subset is used by default.
- ART1 requires binarized inputs, discarding grayscale detail.
- The project is an educational benchmark, not a production system.

---

## References

1. Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image
   Dataset for Benchmarking Machine Learning Algorithms.*
   [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)
2. Fashion-MNIST GitHub repository —
   <https://github.com/zalandoresearch/fashion-mnist>
3. TensorFlow Fashion-MNIST tutorial —
   <https://www.tensorflow.org/tutorials/keras/classification>
4. TensorFlow CNN tutorial —
   <https://www.tensorflow.org/tutorials/images/cnn>
5. Kohonen, T. (1982). *Self-Organized Formation of Topologically Correct
   Feature Maps.* Biological Cybernetics, 43, 59–69.
   [DOI:10.1007/BF00337288](https://doi.org/10.1007/BF00337288)
6. Carpenter, G. A., & Grossberg, S. (1987). *A Massively Parallel Architecture
   for a Self-Organizing Neural Pattern Recognition Machine.* Computer Vision,
   Graphics, and Image Processing, 37(1), 54–115.
7. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning
   Representations by Back-Propagating Errors.* Nature, 323, 533–536.

---

## License

This project is for academic and educational use.
