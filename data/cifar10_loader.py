"""Robust, fast CIFAR-10 data loader that bypasses slow / rate-limited cs.toronto.edu servers
by loading directly from local cache or high-speed CDN mirror.
"""
import os
import sys
import pickle
import tarfile
import urllib.request
import shutil
import numpy as np


def load_cifar10():
    """Loads CIFAR-10 dataset without ever contacting throttled Toronto university servers.

    Returns:
        (x_train, y_train), (x_test, y_test): Uint8 numpy arrays matching Keras cifar10.load_data()
    """
    cache_dir = os.path.expanduser('~/.keras/datasets')
    os.makedirs(cache_dir, exist_ok=True)
    cifar_dir = os.path.join(cache_dir, 'cifar-10-batches-py')
    tar_path = os.path.join(cache_dir, 'cifar-10-batches-py.tar.gz')

    # Verify existing extracted files
    needed_files = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']
    has_extracted = os.path.exists(cifar_dir) and all(
        os.path.exists(os.path.join(cifar_dir, f)) and os.path.getsize(os.path.join(cifar_dir, f)) > 10000000
        for f in needed_files
    )

    if not has_extracted:
        # Check if complete valid tarball already exists (170,498,071 bytes)
        has_tar = os.path.exists(tar_path) and os.path.getsize(tar_path) > 160 * 1024 * 1024

        if not has_tar:
            if os.path.exists(tar_path):
                try:
                    os.remove(tar_path)
                except OSError:
                    pass

            urls = [
                "https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz",
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-batches-py.tar.gz"
            ]

            downloaded = False
            for url in urls:
                try:
                    print(f"Downloading CIFAR-10 from fast CDN: {url} ...")
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=60) as resp, open(tar_path, 'wb') as f:
                        shutil.copyfileobj(resp, f)
                    downloaded = True
                    print("Download complete (~3s)!")
                    break
                except Exception as e:
                    print(f"Mirror failed ({e}), trying fallback...")

            if not downloaded:
                raise RuntimeError("Failed to download CIFAR-10 from fast CDN mirrors.")

        print("Extracting CIFAR-10 dataset into cache...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=cache_dir)
        print("CIFAR-10 extraction complete!")

    # Load 50,000 training images from data_batch_1 .. 5
    x_train_list = []
    y_train_list = []
    for i in range(1, 6):
        batch_path = os.path.join(cifar_dir, f'data_batch_{i}')
        with open(batch_path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            x_train_list.append(d[b'data'])
            y_train_list.append(d[b'labels'])

    x_train = np.concatenate(x_train_list, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.concatenate(y_train_list, axis=0)

    # Load 10,000 test images
    test_path = os.path.join(cifar_dir, 'test_batch')
    with open(test_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        x_test = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(d[b'labels'])

    return (x_train, y_train), (x_test, y_test)


# Monkey-patch Keras so any third-party or legacy call uses the fast loader
try:
    import tensorflow as tf
    tf.keras.datasets.cifar10.load_data = load_cifar10
except Exception:
    pass
