import numpy as np
from PIL import Image
import glob
import os

def load_images_from_folder(
    folder,
    img_size=(94, 125),
    max_images_per_class=None,
    shuffle=True,
    seed=42
):
    """
    Loads cat and dog images from a folder, resizes, and returns as (X, y) numpy arrays.

    Args:
        folder (str): Path to train/ (with cat.*.jpg and dog.*.jpg)
        img_size (tuple): Resize target, e.g., (94, 125)
        max_images_per_class (int): Optionally limit per class (e.g., 2048)
        shuffle (bool): Whether to shuffle data after loading
        seed (int): Random seed for reproducibility

    Returns:
        X: np.ndarray, shape (N, h, w, 3)
        y: np.ndarray, shape (N,)
    """
    # Find all cat and dog images
    cat_files = sorted(glob.glob(os.path.join(folder, 'cat.*.jpg')))
    dog_files = sorted(glob.glob(os.path.join(folder, 'dog.*.jpg')))
    
    if max_images_per_class:
        cat_files = cat_files[:max_images_per_class]
        dog_files = dog_files[:max_images_per_class]

    files = cat_files + dog_files
    labels = [1] * len(cat_files) + [0] * len(dog_files)  # 1=cat, 0=dog

    # Load images
    X = []
    for f in files:
        im = Image.open(f).convert("RGB").resize(img_size)
        X.append(np.array(im))
    X = np.stack(X, axis=0)
    y = np.array(labels, dtype=np.uint8)

    # Shuffle
    if shuffle:
        rng = np.random.RandomState(seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    return X, y

# Example usage:
if __name__ == "__main__":
    X, y = load_images_from_folder('train/', img_size=(94,125), max_images_per_class=2048)
    print("Loaded shape:", X.shape, y.shape)
    print("First 10 labels:", y[:10])
    print("First 10 images shape:", X[:10].shape)