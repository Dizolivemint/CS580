import numpy as np

def train_valid_split(x, y, valid_size=0.1, seed=42, shuffle=True):
    """
    Splits x, y numpy arrays into train and validation sets using a fixed seed.
    """
    assert len(x) == len(y), "Mismatched features/labels length"
    idxs = np.arange(len(x))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idxs)
    x_shuffled = x[idxs]
    y_shuffled = y[idxs]

    split_idx = int(len(x) * (1 - valid_size))
    x_train, x_valid = x_shuffled[:split_idx], x_shuffled[split_idx:]
    y_train, y_valid = y_shuffled[:split_idx], y_shuffled[split_idx:]

    return x_train, x_valid, y_train, y_valid
