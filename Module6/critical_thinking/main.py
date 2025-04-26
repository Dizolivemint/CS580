import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from config import (
    TRAIN_DIR, IMG_SIZE, MAX_IMAGES_PER_CLASS, SEED,
    CONV_FILTERS, CONV2_FILTERS, DENSE_UNITS, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, VALID_SIZE, EARLY_STOPPING, PATIENCE,
    METRIC_THRESHOLD, MODEL_DIR, MODEL_NAME
)
from loader import load_images_from_folder, create_data_loaders
from cnn_model import build_cnn_model
from train import train_model, predict, save_model
from metrics import compute_classification_metrics, find_best_threshold

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading images...")
    X, y = load_images_from_folder(
        TRAIN_DIR,
        img_size=IMG_SIZE,
        max_images_per_class=MAX_IMAGES_PER_CLASS,
        shuffle=True,
        seed=SEED
    )
    print(f"Loaded {len(X)} images.")
    
    # 2. Create data loaders
    print("Creating data loaders...")
    train_loader, valid_loader, train_dataset, valid_dataset = create_data_loaders(
        X, y, batch_size=BATCH_SIZE, valid_size=VALID_SIZE, seed=SEED
    )
    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(valid_dataset)} images")
    
    # 3. Build model
    print("Building model...")
    img_shape = (3, IMG_SIZE[0], IMG_SIZE[1])  # PyTorch expects (C, H, W)
    model, optimizer, criterion = build_cnn_model(
        img_shape,
        conv_filters=CONV_FILTERS,
        conv2_filters=CONV2_FILTERS,
        dense_units=DENSE_UNITS,
        learning_rate=LEARNING_RATE
    )
    
    # 4. Train model
    print("Training model...")
    model, history = train_model(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        valid_loader, 
        epochs=EPOCHS, 
        device=device,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        verbose=1
    )
    
    # 5. Evaluate model
    print("Evaluating model...")
    y_pred_probs, y_valid = predict(model, valid_loader, device)
    
    # Find best threshold for F1
    best_thresh, best_f1 = find_best_threshold(y_valid, y_pred_probs, metric='f1')
    print(f"Best threshold by F1: {best_thresh:.3f} (F1={best_f1:.3f})")
    
    # Compute metrics
    metrics_default = compute_classification_metrics(y_valid, y_pred_probs, threshold=METRIC_THRESHOLD)
    metrics_best = compute_classification_metrics(y_valid, y_pred_probs, threshold=best_thresh)
    
    print("\nMetrics @ Default Threshold (0.5):")
    for k, v in metrics_default.items():
        print(f"{k.capitalize()}: {v:.4f}" if isinstance(v, float) else f"{k.capitalize()}: {v}")
    
    print("\nMetrics @ Best Threshold:")
    for k, v in metrics_best.items():
        print(f"{k.capitalize()}: {v:.4f}" if isinstance(v, float) else f"{k.capitalize()}: {v}")
    
    # 6. Plot scatter plot of predictions vs labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_probs, y=y_valid)
    plt.title("Scatter Plot of Predictions vs Labels")
    plt.xlabel("Predicted Probability (Cat)")
    plt.ylabel("True Label (1=Cat, 0=Dog)")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    plt.savefig(os.path.join(MODEL_DIR, f"{MODEL_NAME}_scatter.png"))
    plt.close()
    
    # 7. Save model
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")
    save_model(model, model_path)
    
    # 8. Save threshold and metrics
    with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_metrics.json"), "w") as f:
        # Convert numpy float32 to native Python float
        metrics_default = {k: float(v) if isinstance(v, np.float32) else v for k, v in metrics_default.items()}
        metrics_best = {k: float(v) if isinstance(v, np.float32) else v for k, v in metrics_best.items()}
        
        combined_metrics = {
            'default_threshold': metrics_default,
            'best_threshold': metrics_best
        }
        json.dump(combined_metrics, f, indent=4)
    
    print("Evaluation complete!")
    
    # 9. Create an interface to visualize predictions
    # This is implemented in the use_model.py file
    print("Use the interface in use_model.py to visualize predictions!")

if __name__ == "__main__":
    main()