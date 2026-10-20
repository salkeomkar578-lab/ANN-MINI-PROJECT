"""
dataset_loader.py — Official Zalando Fashion-MNIST Dataset Loader

Reference Repository:
https://github.com/zalandoresearch/fashion-mnist

This module provides native loading utilities based on the official Zalando
Research Python loader (mnist_reader.py), with zero deep-learning dependencies,
supporting both locally downloaded gz files and cached instances.
"""

import os
import gzip
import urllib.request
import numpy as np

# Official Zalando Fashion-MNIST GitHub raw URL base
ZALANDO_RAW_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

FILENAMES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz',
}


def load_mnist_zalando(data_dir=None, kind='train'):
    """
    Load Fashion-MNIST directly from raw idx-ubyte.gz files.
    Matches the official Zalando Research reader specification:
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

    Parameters
    ----------
    data_dir : str, optional
        Directory where gz files are stored. If None, checks local './data'
        or default Keras cache directory (~/.keras/datasets/fashion-mnist).
    kind : str, default 'train'
        Either 'train' or 't10k' (test).

    Returns
    -------
    images : ndarray of shape (N, 784), dtype=uint8
    labels : ndarray of shape (N,), dtype=uint8
    """
    if data_dir is None:
        keras_cache = os.path.expanduser(os.path.join('~', '.keras', 'datasets', 'fashion-mnist'))
        if os.path.exists(keras_cache):
            data_dir = keras_cache
        else:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')

    labels_path = os.path.join(data_dir, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(data_dir, f'{kind}-images-idx3-ubyte.gz')

    if not os.path.exists(labels_path) or not os.path.exists(images_path):
        raise FileNotFoundError(
            f"Fashion-MNIST files not found in {data_dir}.\n"
            f"Expected: {labels_path} and {images_path}\n"
            f"You can download them from https://github.com/zalandoresearch/fashion-mnist"
        )

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def download_zalando_dataset(target_dir='data'):
    """
    Download the raw official Zalando Fashion-MNIST files directly from
    the official repository hosting mirror into target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading official Zalando Fashion-MNIST to '{target_dir}'...")

    for key, filename in FILENAMES.items():
        file_path = os.path.join(target_dir, filename)
        if not os.path.exists(file_path):
            url = ZALANDO_RAW_URL + filename
            print(f"  Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"  Saved {filename} ({os.path.getsize(file_path):,} bytes)")
        else:
            print(f"  {filename} already exists ({os.path.getsize(file_path):,} bytes).")

    print("All Zalando Fashion-MNIST files ready.")


def load_fashion_mnist_all(normalize=True):
    """
    Convenience loader returning both train and test splits,
    compatible with all four ANN models.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        If normalize=True, X arrays are float32 in [0, 1].
        If normalize=False, X arrays are uint8 in [0, 255].
    """
    try:
        # First attempt: load via TensorFlow / Keras cache
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    except Exception:
        # Fallback: load directly using Zalando gzip reader
        X_train, y_train = load_mnist_zalando(kind='train')
        X_test, y_test = load_mnist_zalando(kind='t10k')

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    print("Testing Zalando Fashion-MNIST Loader...")
    (X_tr, y_tr), (X_te, y_te) = load_fashion_mnist_all(normalize=True)
    print(f"Train samples: {X_tr.shape[0]} | Shape: {X_tr.shape} | Range: [{X_tr.min():.1f}, {X_tr.max():.1f}]")
    print(f"Test samples : {X_te.shape[0]} | Shape: {X_te.shape} | Range: [{X_te.min():.1f}, {X_te.max():.1f}]")
    print("Classes      :", CLASS_NAMES)
    print("[OK] Zalando Fashion-MNIST Loader verified successfully.")
