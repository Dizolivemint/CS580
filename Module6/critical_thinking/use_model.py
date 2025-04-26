import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

from config import IMG_SIZE, MODEL_DIR, MODEL_NAME, TRAIN_DIR, SEED
from cnn_model import build_cnn_model
from loader import CatDogDataset, load_images_from_folder
from torch.utils.data import DataLoader

def load_model(model_path, img_shape=(3, 94, 125), conv_filters=24, conv2_filters=48, dense_units=128):
    """
    Load a trained PyTorch model.
    
    Args:
        model_path: Path to the saved model
        img_shape: Shape of input images (C, H, W)
        conv_filters: Number of filters in first conv layer
        conv2_filters: Number of filters in second conv layer
        dense_units: Number of units in dense layers
        
    Returns:
        model: Loaded PyTorch model
    """
    # Create the model
    model, _, _ = build_cnn_model(
        img_shape, 
        conv_filters=conv_filters, 
        conv2_filters=conv2_filters, 
        dense_units=dense_units
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def predict_image(model, image_path=None, image_data=None):
    """
    Predict whether an image contains a cat or dog.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image file
        image_data: Alternatively, provide image data directly
        
    Returns:
        probability: Probability that the image contains a cat
    """
    # Load and preprocess image
    if image_path:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
    elif image_data is not None:
        img_array = image_data
    else:
        raise ValueError("Either image_path or image_data must be provided")
    
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Get probability
    probability = prediction.item()
    
    return probability

def visualize_prediction(model, x_valid, index):
    """
    Visualize the prediction for a specific image.
    
    Args:
        model: Trained PyTorch model
        x_valid: Validation data
        index: Index of the image to visualize
        
    Returns:
        probability: Probability that the image contains a cat
    """
    # Get the image
    img = x_valid[index]
    
    # Make prediction
    probability = predict_image(model, image_data=img)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Probability of being a cat: {probability:.4f}")
    plt.axis('off')
    plt.show()
    
    return probability

def interactive_interface(model, x_valid, labels_valid=None):
    """
    Interactive interface to visualize predictions.
    
    Args:
        model: Trained PyTorch model
        x_valid: Validation data
        labels_valid: Validation labels (optional)
    """
    print("\n=== Cat vs Dog Classifier Interface ===")
    print("This interface allows you to visualize the model's predictions.")
    
    while True:
        # Get user input
        try:
            index = int(input(f"\nEnter an index (0-{len(x_valid)-1}) to view prediction, or -1 to quit: "))
            if index == -1:
                break
            if index < 0 or index >= len(x_valid):
                print(f"Invalid index. Please enter a value between 0 and {len(x_valid)-1}")
                continue
        except ValueError:
            print("Please enter a valid integer.")
            continue
        
        # Make prediction
        probability = predict_image(model, image_data=x_valid[index])
        
        # Display image
        img = Image.fromarray(x_valid[index])
        img.show()
        
        # Print results
        print(f"Probability of being a cat: {probability:.4f}")
        print(f"Predicted class: {'Cat' if probability > 0.5 else 'Dog'}")
        
        if labels_valid is not None:
            print(f"True class: {'Cat' if labels_valid[index] == 1 else 'Dog'}")
            print(f"Prediction is {'correct' if (probability > 0.5) == (labels_valid[index] == 1) else 'incorrect'}")
    
    print("Exiting interface.")

def main():
    """Main function to run the interface."""
    parser = argparse.ArgumentParser(description='Cat vs Dog Classifier Interface')
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt"),
                       help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default=TRAIN_DIR,
                       help='Path to the data directory')
    args = parser.parse_args()
    
    # Load the model
    print("Loading model...")
    model = load_model(args.model_path)
    
    # Load a few validation images for testing
    print("Loading validation data...")
    X, y = load_images_from_folder(
        args.data_dir,
        img_size=IMG_SIZE,
        max_images_per_class=512,  # Smaller set for faster loading
        shuffle=True,
        seed=SEED
    )
    
    # Run the interface
    interactive_interface(model, X, y)

if __name__ == "__main__":
    main()