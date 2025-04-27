import gradio as gr
import numpy as np
import torch
from PIL import Image

from config import IMG_SIZE, MODEL_DIR, MODEL_NAME, TRAIN_DIR, SEED
from cnn_model import build_cnn_model
from loader import load_images_from_folder

# Load validation data once
print("Loading validation data...")
X_valid, y_valid = load_images_from_folder(
    TRAIN_DIR,
    img_size=IMG_SIZE,
    max_images_per_class=512,  # Or your valid size
    shuffle=True,
    seed=SEED
)

# Load model once
def load_model(model_path, img_shape=(3, 94, 125), conv_filters=24, conv2_filters=48, dense_units=128):
    model, _, _ = build_cnn_model(
        img_shape,
        conv_filters=conv_filters,
        conv2_filters=conv2_filters,
        dense_units=dense_units
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model_path = f"{MODEL_DIR}/{MODEL_NAME}.pt"
model = load_model(model_path)

# Prediction function
def predict_index(index):
    img = X_valid[index]
    img_disp = Image.fromarray(img)

    # Prepare for model
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        prob = model(img_tensor).item()

    pred_class = "Cat" if prob > 0.5 else "Dog"
    true_class = "Cat" if y_valid[index] == 1 else "Dog"
    correct = "✅ Correct" if pred_class == true_class else "❌ Incorrect"

    prob_display = f"{prob:.4f}"

    return img_disp, pred_class, prob_display, true_class, correct

# Gradio interface
demo = gr.Interface(
    fn=predict_index,
    inputs=gr.Slider(0, len(X_valid)-1, step=1, label="Image Index"),
    outputs=[
        gr.Image(label="Selected Image"),
        gr.Text(label="Predicted Class"),
        gr.Text(label="Probability of Cat"),
        gr.Text(label="True Class"),
        gr.Text(label="Correct?"),
    ],
    title="Cat vs Dog Classifier",
    description="Choose an image index to see the model's prediction, probability, and true class."
)

if __name__ == "__main__":
    demo.launch()
