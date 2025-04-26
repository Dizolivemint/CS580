import numpy as np
from PIL import Image
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader

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

class CatDogDataset(Dataset):
    """PyTorch Dataset for Cat vs Dog classification."""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: Numpy array of images
            labels: Numpy array of labels
            transform: Optional transform to apply to images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert image to PyTorch tensor (HWC -> CHW)
        if self.transform:
            image = self.transform(image)
        else:
            # Default conversion to PyTorch tensor and normalization
            image = torch.from_numpy(image).float()
            # Rearrange from HWC to CHW (PyTorch convention)
            image = image.permute(2, 0, 1)
            # Normalize to [0, 1]
            image = image / 255.0
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label

def create_data_loaders(X, y, batch_size=32, valid_size=0.1, seed=42):
    """
    Creates PyTorch DataLoaders for training and validation.
    
    Args:
        X: Images as numpy array
        y: Labels as numpy array
        batch_size: Batch size for training
        valid_size: Proportion of data to use for validation
        seed: Random seed
        
    Returns:
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        train_dataset: Dataset for training
        valid_dataset: Dataset for validation
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Split into train and validation sets
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - valid_size))
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]
    
    # Create datasets
    train_dataset = CatDogDataset(
        X[train_indices], 
        y[train_indices]
    )
    
    valid_dataset = CatDogDataset(
        X[valid_indices], 
        y[valid_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, train_dataset, valid_dataset

# Example usage:
if __name__ == "__main__":
    X, y = load_images_from_folder('train/', img_size=(94,125), max_images_per_class=2048)
    print("Loaded shape:", X.shape, y.shape)
    print("First 10 labels:", y[:10])
    
    train_loader, valid_loader, _, _ = create_data_loaders(X, y, batch_size=32)
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(valid_loader)}")